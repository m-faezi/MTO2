import numpy as np
from scipy import stats
from mto2lib.reduction import morphological_background as mb


_reject_tile = False
_accept_tile = True
rejection_rate_1 = 0
rejection_rate_2 = 0


def estimate_background(img, rejection_rate=0.05):

    global rejection_rate_1, rejection_rate_2

    rejection_rate_1 = 1 - pow(1 - rejection_rate, 0.5)
    rejection_rate_2 = 1 - pow(1 - rejection_rate, 0.25)
    
    tile_size = largest_flat_tile(img)
    bg_mean, bg_var, gain = collect_info(img, tile_size)
    background_map = np.full_like(img, bg_mean, dtype=np.float32)

    return bg_mean, bg_var, gain, background_map


def largest_flat_tile(img, tile_size_start=6, tile_size_min=4, tile_size_max=7):

    current_size = 2 ** tile_size_start
    max_size = 2 ** tile_size_max
    min_size = 2 ** tile_size_min

    if available_tiles(img, current_size):

        while current_size < max_size:

            current_size *= 2

            if not available_tiles(img, current_size):

                return int(current_size / 2)

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

    if not test_mean_equality(tile[:, :half_width], tile[:, half_width:], sig_level):

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
    gain = np.where(bg_var != 0, (bg_mean - soft_bias) / np.float32(bg_var), 0)

    return bg_mean, bg_var, gain


def replace_nans(img, value=np.inf):

    if value == 0:

        return np.nan_to_num(img)

    else:
        img[np.isnan(img)] = value

        return img

