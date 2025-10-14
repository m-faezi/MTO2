import numpy as np
import higra as hg
from mto2lib.utils import base_utils as uts
import os


def extract_parameters(
        image,
        header,
        tree_of_segments,
        n_map_segments,
        parent_altitude,
        area,
        unique_segment_ids,
        arguments,
):

    print("Extracting parameters...")

    segment_ids = np.arange(tree_of_segments.num_leaves(), tree_of_segments.num_vertices())
    label_data = np.full(tree_of_segments.num_vertices(), -1, dtype=np.int32)
    label_data[segment_ids] = np.arange(len(segment_ids))
    seg_array = hg.reconstruct_leaf_data(tree_of_segments, label_data)

    coords_per_segment = [[] for _ in range(len(segment_ids))]

    for y_ in range(seg_array.shape[0]):

        for x_ in range(seg_array.shape[1]):

            label = seg_array[y_, x_]

            if label >= 0:

                coords_per_segment[label].append((y_, x_))

    r_eff = [uts.half_light_radius(image, coords) for coords in coords_per_segment]
    r_fwhm = [uts.compute_r_fwhm(image, coords) for coords in coords_per_segment]

    centroids = [uts.weighted_centroid_coords_from_segments(image, coords) for coords in coords_per_segment]

    y = [cen[0] for cen in centroids]
    x = [cen[1] for cen in centroids]

    ra, dec = uts.sky_coordinates(y, x, header)

    a, b, theta = uts.second_order_moments(tree_of_segments, image.shape[:2], image)
    flux = hg.accumulate_sequential(tree_of_segments, image, hg.Accumulators.sum)

    results_dir = os.path.join("./results", arguments.time_stamp)
    output_csv = os.path.join(results_dir, "parameters.csv")

    uts.save_parameters(
        unique_segment_ids[tree_of_segments.num_leaves():][::-1],
        x[::-1],
        y[::-1],
        ra[::-1],
        dec[::-1],
        flux[tree_of_segments.num_leaves():][::-1],
        flux[tree_of_segments.num_leaves():][::-1] -
        parent_altitude[n_map_segments][tree_of_segments.num_leaves():][::-1],
        area[n_map_segments][tree_of_segments.num_leaves():][::-1],
        a[tree_of_segments.num_leaves():][::-1],
        b[tree_of_segments.num_leaves():][::-1],
        theta[tree_of_segments.num_leaves():][::-1],
        r_eff[::-1],
        r_fwhm[::-1],
        file_name=output_csv
    )

    return None

