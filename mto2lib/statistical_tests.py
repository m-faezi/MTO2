import numpy as np
from scipy.stats import chi2
import higra as hg


def attribute_statistical_significance(tree, altitudes, volume, area, background_var, gain, alpha=1e-6):

    denominator = background_var + altitudes[tree.parents()] / (gain + np.finfo(np.float64).eps)
    safe_denominator = np.where(denominator == 0, np.finfo(np.float64).eps, denominator)
    volume /= safe_denominator

    significant_nodes = volume > chi2.ppf(alpha, area)
    significant_nodes[:tree.num_leaves()] = False

    return significant_nodes


def attribute_main_branch(tree):

    area = hg.attribute_area(tree)
    largest_child = hg.accumulate_parallel(tree, area, hg.Accumulators.argmax)
    child_number = hg.attribute_child_number(tree)

    return child_number == largest_child[tree.parents()]


def select_objects(tree, significant_nodes):

    filtered_tree, node_map = hg.simplify_tree(tree, np.logical_not(significant_nodes))
    main_branch = attribute_main_branch(filtered_tree)

    if not significant_nodes[tree.root()]:
        root_children = filtered_tree.children(filtered_tree.root())
        main_branch[root_children] = False

    res = np.zeros(tree.num_vertices(), dtype=np.bool_)
    res[node_map] = np.logical_not(main_branch)

    return res


def move_up(
    tree, altitudes, area, parent_area, distances, objects, background_var, gain, gamma_distance, gaussian,
    move_factor, G_fit, area_ratio
):

    main_branch = attribute_main_branch(tree)

    closest_object_ancestor = hg.propagate_sequential(
        tree,
        np.arange(tree.num_vertices()),
        np.logical_and(main_branch, np.logical_not(objects)))

    target_altitudes = altitudes.copy()
    object_indexes, = np.nonzero(objects)
    local_noise = np.sqrt(
        np.maximum(np.where(gain != 0, background_var + altitudes[tree.parent(object_indexes)], 0) / gain, 0)
    )
    target_altitudes[object_indexes] = altitudes[object_indexes] + move_factor * local_noise

    target_altitudes = target_altitudes[closest_object_ancestor]

    if not G_fit:

        valid_moves = np.logical_and(
            altitudes >= target_altitudes,
            np.logical_and(
                objects[closest_object_ancestor],
                area/parent_area >= area_ratio
            )
        )

    elif G_fit:

        valid_moves = np.logical_and(
            np.logical_and(
                altitudes >= target_altitudes,
                np.logical_and(
                    objects[closest_object_ancestor],
                    area/parent_area >= area_ratio
                )
            ),
            altitudes>=gaussian
        )

    parent_closest_object_ancestor = closest_object_ancestor[tree.parents()]
    parent_not_valid_moves = np.logical_not(valid_moves[tree.parents()])
    new_objects = np.logical_and(
        valid_moves,
        np.logical_or(
            parent_not_valid_moves,
            parent_closest_object_ancestor != closest_object_ancestor
        )
    )

    return new_objects

