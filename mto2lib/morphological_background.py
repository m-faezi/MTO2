import mto2lib.utils as uts
import higra as hg
import numpy as np


def estimate_structural_background(image, return_map=False):

    graph_structure, tree_structure, altitudes = uts.image_to_hierarchical_structure(image)

    x, y = uts.centroid(tree_structure, image.shape[:2])
    distances = np.sqrt((x[tree_structure.parents()] - x) ** 2 + (y[tree_structure.parents()] - y) ** 2)

    area, volume, mean, variance, topological_height, distance_to_root_center = uts.get_max_tree_attributes(
        tree_structure, altitudes, image
    )

    non_bool_unique_topological_height = uts.mark_non_unique(topological_height)
    masked_topological_height = topological_height[~non_bool_unique_topological_height]
    masked_area = area[~non_bool_unique_topological_height]
    masked_volume = volume[~non_bool_unique_topological_height]
    masked_altitudes = altitudes[~non_bool_unique_topological_height]
    masked_distance_to_root = distance_to_root_center[~non_bool_unique_topological_height]

    features = [
        masked_topological_height,
        masked_area,
        masked_volume,
        masked_altitudes,
        masked_distance_to_root
    ]

    filtered_features = [feature for feature in features if not np.isnan(feature).any()]

    if not filtered_features:

        gaussian_intensities = uts.compute_gaussian_profile(
            mean,
            variance,
            distances,
            altitudes / area
        )

        tree_non_source, n_map_non_source = hg.simplify_tree(
            tree_structure,
            np.logical_or(
                ~non_bool_unique_topological_height,
                altitudes / area >= gaussian_intensities
            )
        )

        morph_background = hg.reconstruct_leaf_data(tree_non_source, altitudes[n_map_non_source])

    else:

        all_labels = uts.binary_cluster_bg_structure_minibatch(
            filtered_features,
            non_bool_unique_topological_height,
            altitudes
        )

        gaussian_intensities = uts.compute_gaussian_profile(
            mean,
            variance,
            distances,
            altitudes/area
        )

        tree_non_source, n_map_non_source = hg.simplify_tree(
            tree_structure,
            np.logical_or(
                all_labels != all_labels[tree_structure.root()],
                altitudes / area >= gaussian_intensities
            )
        )

        morph_background = hg.reconstruct_leaf_data(tree_non_source, altitudes[n_map_non_source])

    morph_background_map = np.full_like(image, morph_background, dtype=np.float64)

    soft_bias = 0.0
    image_minimum = np.nanmin(image)

    if image_minimum < 0:

        soft_bias = image_minimum

    bg_mean = np.nanmean(morph_background, axis=None)
    bg_var = np.nanvar(morph_background, axis=None)
    gain = (bg_mean - np.abs(soft_bias)) / np.maximum(bg_var, np.finfo(np.float64).eps)

    if return_map:

        return morph_background_map, bg_var, gain, morph_background_map

    else:

        return bg_mean, bg_var, gain


