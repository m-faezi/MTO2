from mto2lib.utils import base_utils as uts
from mto2lib.utils import ml_utils as ml_uts
from mto2lib.utils import torch_utils as tc_uts
import higra as hg
import numpy as np


def estimate_structural_background(image, maxtree):

    main_branch = uts.get_main_branch(maxtree.gamma)

    mb_topological_height = maxtree.gamma[main_branch]
    mb_area = maxtree.area[main_branch]
    mb_volume = maxtree.volume[main_branch]
    mb_altitudes = maxtree.altitudes[main_branch]
    # mb_distance_to_root = maxtree.distance_to_root_center[main_branch]

    features = [
        mb_topological_height,
        np.log10(mb_area),
        np.log10(mb_volume),
        np.log10(mb_altitudes),
        # mb_distance_to_root
    ]

    filtered_features = [feature for feature in features if not np.isnan(feature).any()]

    try:

        try:

            all_labels = tc_uts.pytorch_fuzzy_c_means(
                filtered_features,
                ~main_branch,
                maxtree.altitudes
            )

        except Exception as e:

            all_labels = ml_uts.fuzz_bg_structure(
                filtered_features,
                ~main_branch,
                maxtree.altitudes
            )

        unique_labels = np.unique(all_labels)

        if len(unique_labels) == 2:

            area_label_0 = np.mean(maxtree.area[all_labels == unique_labels[0]])
            area_label_1 = np.mean(maxtree.area[all_labels == unique_labels[1]])

            if area_label_0 > area_label_1:

                keep_label = unique_labels[0]
                mean_area = area_label_0

            else:

                keep_label = unique_labels[1]
                mean_area = area_label_1

            all_labels[maxtree.area > mean_area] = keep_label

        tree_non_source, n_map_non_source = hg.simplify_tree(
            maxtree.tree_structure,
            np.logical_or(
                all_labels != keep_label,
                maxtree.area < mean_area
            )
        )

        morph_background = hg.reconstruct_leaf_data(tree_non_source, maxtree.altitudes[n_map_non_source])


    except Exception as e:

        tree_non_source, n_map_non_source = hg.simplify_tree(
            maxtree.tree_structure,
            np.logical_or(
                main_branch,
                maxtree.altitudes / maxtree.area >= maxtree.gaussian_intensities
            )
        )

        morph_background = hg.reconstruct_leaf_data(tree_non_source, maxtree.altitudes[n_map_non_source])



    morph_background_map = np.full_like(image, morph_background, dtype=np.float32)

    soft_bias = 0.0
    image_minimum = np.nanmin(image)

    if image_minimum < 0:

        soft_bias = image_minimum

    bg_mean = np.nanmean(morph_background, axis=None)
    bg_var = np.nanvar(morph_background, axis=None)
    gain = (bg_mean - np.abs(soft_bias)) / np.maximum(bg_var, np.finfo(np.float32).eps)

    return bg_mean, bg_var, gain, morph_background_map

