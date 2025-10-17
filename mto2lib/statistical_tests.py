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


def select_objects(maxtree):

    filtered_tree, node_map = hg.simplify_tree(maxtree.tree_structure, np.logical_not(maxtree.significant_nodes))
    main_branch_local = attribute_main_branch(filtered_tree, maxtree.area[node_map])

    if not maxtree.significant_nodes[maxtree.tree_structure.root()]:
        root_children = filtered_tree.children(filtered_tree.root())
        main_branch_local[root_children] = False

    res = np.zeros(maxtree.tree_structure.num_vertices(), dtype=np.bool_)
    res[node_map] = np.logical_not(main_branch_local)

    return res


