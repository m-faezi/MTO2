import helper
import higra as hg
import numpy as np
from scipy import stats


_reject_tile = False
_accept_tile = True
rejection_rate_1 = 0
rejection_rate_2 = 0


def estimate_background(img, rejection_rate=0.05):

    global rejection_rate_1, rejection_rate_2
    rejection_rate_1 = 1 - pow(1 - rejection_rate, 0.5)
    rejection_rate_2 = 1 - pow(1 - rejection_rate, 0.25)
    tile_size = largest_flat_tile(img)

    if tile_size == 0:

        return estimate_structural_background(img)

    return collect_info(img, tile_size)


def largest_flat_tile(img, tile_size_start=6, tile_size_min=4, tile_size_max=7):

    current_size = 2**tile_size_start
    max_size = 2**tile_size_max
    min_size = 2**tile_size_min

    if available_tiles(img, current_size):

        while current_size < max_size:
            current_size *= 2
            if not available_tiles(img, current_size):

                return int(current_size/2)

        return max_size

    else:

        while current_size > min_size:
            current_size = int(current_size / 2)
            if available_tiles(img, current_size):

                return min_size

    return 0


def available_tiles(img, tile_length):

    for y in range(0, img.shape[0] - tile_length, tile_length):
        for x in range(0, img.shape[1] - tile_length, tile_length):
            if check_tile_is_flat(img[y: y + tile_length, x: x + tile_length]):

                return True

    return False


def collect_info(img, tile_length):

    flat_tiles = []

    for y in range(0, img.shape[0] - tile_length, tile_length):
        for x in range(0, img.shape[1] - tile_length, tile_length):
            if check_tile_is_flat(img[y: y + tile_length, x: x + tile_length]):
                flat_tiles.append([x, y])

    return est_mean_and_variance_gain(img, tile_length, flat_tiles)


def check_tile_is_flat(tile):

    if np.all(tile == 0):

        return _reject_tile

    if np.count_nonzero(~np.isnan(tile)) == 0:

        return _reject_tile

    if test_normality(tile, rejection_rate_1) is False:

        return _reject_tile

    if check_tile_means(tile, rejection_rate_2) is False:

        return _reject_tile

    return _accept_tile


def check_tile_means(tile, sig_level):

    half_height = int(tile.shape[0] / 2)
    half_width = int(tile.shape[1] / 2)

    if not test_mean_equality(tile[:half_height, :], tile[half_height:, :], sig_level):

        return _reject_tile

    if not test_mean_equality(
            tile[:, :half_width], tile[:, half_width:], sig_level):

        return _reject_tile

    return _accept_tile


def test_normality(array, test_statistic):

    k2, p = stats.normaltest(array.ravel(), nan_policy='omit')

    if p < test_statistic:

        return _reject_tile

    else:

        return _accept_tile


def test_mean_equality(array_a, array_b, test_statistic):

    s, p = stats.ttest_ind(array_a.ravel(), array_b.ravel(), nan_policy='omit')

    if p < test_statistic:

        return _reject_tile

    else:

        return _accept_tile


def est_mean_and_variance_gain(img, tile_length, usable):

    total_bg = np.vstack(
        [img[u[1]: u[1] + tile_length, u[0]: u[0] + tile_length] for u in usable]
    )

    soft_bias = 0.0
    image_minimum = np.nanmin(img)

    if image_minimum < 0:
        soft_bias = image_minimum

    bg_mean = np.nanmean(total_bg, axis=None)
    bg_var = np.nanvar(total_bg, axis=None)
    gain = np.where(bg_var != 0, (bg_mean - soft_bias) / np.float16(bg_var), 0)

    return bg_mean, bg_var, gain


def replace_nans(img, value=np.inf):

    if value == 0:

        return np.nan_to_num(img)

    else:
        img[np.isnan(img)] = value

        return img


def estimate_structural_background(image):

    graph_structure, tree_structure, altitudes = helper.image_to_hierarchical_structure(image)

    x, y = helper.centroid(tree_structure, image.shape[:2])
    distances = np.sqrt((x[tree_structure.parents()] - x) ** 2 + (y[tree_structure.parents()] - y) ** 2)

    area, volume, mean, variance, topological_height, distance_to_root_center = helper.get_max_tree_attributes(
        tree_structure, altitudes, image
    )

    non_bool_unique_topological_height = helper.mark_non_unique(topological_height)
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

        gaussian_intensities = helper.compute_gaussian_profile(
            mean,
            variance,
            distances,
            altitudes / area
        )

        tree_non_source, n_map_non_source = hg.simplify_tree(
            tree_structure,
            np.logical_or(
                ~non_bool_unique_topological_height,
                altitudes / area <= gaussian_intensities
            )
        )

        background = hg.reconstruct_leaf_data(tree_non_source, altitudes[n_map_non_source])

    else:

        all_labels = helper.fuzz_bg_structure(
            filtered_features,
            non_bool_unique_topological_height,
            altitudes
        )

        gaussian_intensities = helper.compute_gaussian_profile(
            mean,
            variance,
            distances,
            altitudes/area
        )

        tree_non_source, n_map_non_source = hg.simplify_tree(
            tree_structure,
            np.logical_or(
                all_labels != all_labels[tree_structure.root()],
                altitudes/area <= gaussian_intensities
            )
        )

        background = hg.reconstruct_leaf_data(tree_non_source, altitudes[n_map_non_source])

    soft_bias = 0.0
    image_minimum = np.nanmin(image)

    if image_minimum < 0:
        soft_bias = image_minimum

    bg_mean = np.nanmean(background, axis=None)
    bg_var = np.nanvar(background, axis=None)
    gain = (bg_mean - np.abs(soft_bias)) / np.maximum(bg_var, np.finfo(np.float64).eps)

    return bg_mean, bg_var, gain


