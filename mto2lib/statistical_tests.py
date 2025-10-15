import numpy as np
from scipy.stats import chi2
import higra as hg


def attribute_statistical_significance(maxtree, dark_frame, alpha=1e-6):

    denominator = (dark_frame.bg_var + maxtree.altitudes[maxtree.tree_structure.parents()] /
                   (dark_frame.bg_gain + np.finfo(np.float32).eps))

    safe_denominator = np.where(denominator == 0, np.finfo(np.float32).eps, denominator)
    maxtree.volume /= safe_denominator

    significant_nodes = maxtree.volume > chi2.ppf(alpha, maxtree.area)
    significant_nodes[:maxtree.tree_structure.num_leaves()] = False

    return significant_nodes


def attribute_main_branch(tree_structure, area):

    largest_child = hg.accumulate_parallel(tree_structure, area, hg.Accumulators.argmax)
    child_number = hg.attribute_child_number(tree_structure)

    return child_number == largest_child[tree_structure.parents()]


def select_objects(maxtree, significant_nodes):

    filtered_tree, node_map = hg.simplify_tree(maxtree.tree_structure, np.logical_not(significant_nodes))
    main_branch_local = attribute_main_branch(filtered_tree, maxtree.area[node_map])

    if not significant_nodes[maxtree.tree_structure.root()]:
        root_children = filtered_tree.children(filtered_tree.root())
        main_branch_local[root_children] = False

    res = np.zeros(maxtree.tree_structure.num_vertices(), dtype=np.bool_)
    res[node_map] = np.logical_not(main_branch_local)

    return res


def move_up(
        maxtree,
        dark_frame,
        run
):

    main_branch_local = attribute_main_branch(maxtree.tree_structure, maxtree.area)

    closest_object_ancestor = hg.propagate_sequential(
        maxtree.tree_structure,
        np.arange(maxtree.tree_structure.num_vertices()),
        np.logical_and(main_branch_local, np.logical_not(maxtree.init_segments)))

    target_altitudes = maxtree.altitudes.copy()
    object_indexes, = np.nonzero(maxtree.init_segments)
    local_noise = np.sqrt(
        np.maximum(
            np.where(
                dark_frame.bg_gain != 0, dark_frame.bg_var + maxtree.altitudes[maxtree.tree_structure.parent(object_indexes)],
                0
            ) / dark_frame.bg_gain,
            0
        )
    )

    target_altitudes[object_indexes] = maxtree.altitudes[object_indexes] + run.arguments.move_factor * local_noise

    target_altitudes = target_altitudes[closest_object_ancestor]

    if not run.arguments.G_fit:

        valid_moves = np.logical_and(
            maxtree.altitudes >= target_altitudes,
            np.logical_and(
                maxtree.init_segments[closest_object_ancestor],
                maxtree.area/maxtree.parent_area >= run.arguments.area_ratio
            )
        )

    elif run.arguments.G_fit:

        valid_moves = np.logical_and(
            np.logical_and(
                maxtree.altitudes >= target_altitudes,
                np.logical_and(
                    maxtree.objects[closest_object_ancestor],
                    maxtree.area/maxtree.parent_area >= run.arguments.area_ratio
                )
            ),
            maxtree.altitudes/maxtree.area>=maxtree.gaussian
        )

    parent_closest_object_ancestor = closest_object_ancestor[maxtree.tree_structure.parents()]
    parent_not_valid_moves = np.logical_not(valid_moves[maxtree.tree_structure.parents()])
    new_objects = np.logical_and(
        valid_moves,
        np.logical_or(
            parent_not_valid_moves,
            parent_closest_object_ancestor != closest_object_ancestor
        )
    )

    return new_objects

