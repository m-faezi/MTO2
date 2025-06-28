import os
import helper
import argparse
import background
import numpy as np
import higra as hg
from PIL import Image, ImageOps


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Path to the image file')
    parser.add_argument(
        '--move_factor',
        type=helper.restricted_non_negative,
        default=0,
        help='move_factor parameter for isophote correction (default = 0)'
    )
    parser.add_argument(
        '--area_ratio',
        type=helper.restricted_normal,
        default=0.90,
        help='area_ratio parameter for deblending correction (default = .90)'
    )
    parser.add_argument(
        '--par_out',
        action='store_true',
        help='Extract and save parameters, if set'
    )
    parser.add_argument(
        '--s_sigma',
        type=helper.restricted_non_negative,
        default=2,
        help='Standard deviation for smoothing Gaussian kernel'
    )
    parser.add_argument(
        '--G_fit',
        action='store_true',
        help='Applies morphological Gaussian filter'
    )
    parser.add_argument(
        '--reduce',
        action='store_true',
        help='Returns background subtracted image'
    )
    parser.add_argument(
        '--file_tag',
        type=str,
        default='',
        help='Optional string to append to saved file names'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='.',
        help='Directory to save output files (default: current directory)'
    )

    args = parser.parse_args()
    data_path = args.file_path
    move_factor = args.move_factor
    area_ratio = args.area_ratio
    par_out = args.par_out
    G_fit = args.G_fit
    s_sigma = args.s_sigma
    reduce = args.reduce
    file_tag = f"-{args.file_tag}" if args.file_tag else ""
    output_path = os.path.abspath(args.output_path)
    os.makedirs(output_path, exist_ok=True)

    image, header = helper.read_image_data(data_path)
    image = helper.image_value_check(image)
    image_processed = helper.smooth_filter(image, s_sigma)

    bg_mean, bg_var, bg_gain = background.estimate_background(image_processed)
    image_calibrated = image_processed - bg_mean

    graph_structure, tree_structure, altitudes = helper.image_to_hierarchical_structure(image_calibrated)

    x, y = helper.centroid(tree_structure, image.shape[:2])
    distances = np.sqrt((x[tree_structure.parents()] - x) ** 2 + (y[tree_structure.parents()] - y) ** 2)
    mean, variance = hg.attribute_gaussian_region_weights_model(tree_structure, image_calibrated)
    area = hg.attribute_area(tree_structure)
    parent_area = area[tree_structure.parents()]

    gaussian_intensities = helper.compute_gaussian_profile(
        mean,
        variance,
        distances,
        altitudes/area
    )

    volume = hg.attribute_volume(tree_structure, altitudes)
    parent_altitude = altitudes[tree_structure.parents()]
    gamma = hg.attribute_topological_height(tree_structure)
    parent_gamma = gamma[tree_structure.parents()]

    significant_nodes = helper.attribute_statistical_significance(
        tree_structure, altitudes, volume, area, bg_var, bg_gain
    )
    objects = helper.select_objects(tree_structure, significant_nodes)

    modified_isophote = helper.move_up(
        tree_structure, altitudes, area, parent_area, distances, objects, bg_var, bg_gain,
        parent_gamma - gamma, gaussian_intensities, move_factor, G_fit, area_ratio
    )

    tree_of_segments, n_map_segments = hg.simplify_tree(tree_structure, np.logical_not(modified_isophote))

    colors = np.random.randint(0, 254, (tree_of_segments.num_vertices(), 3), dtype=np.uint8)
    colors[tree_of_segments.root(), :] = 0
    seg = hg.reconstruct_leaf_data(tree_of_segments, colors)

    segmentation_image = Image.fromarray(seg.astype(np.uint8))
    segmentation_image = ImageOps.flip(segmentation_image)
    unique_segment_ids = np.arange(tree_of_segments.num_vertices())[::-1]
    seg_with_ids = hg.reconstruct_leaf_data(tree_of_segments, unique_segment_ids)

    move_factor_str = str(move_factor).replace('.', 'p')
    area_ratio_str = str(area_ratio).replace('.', 'p')
    s_sigma_str = str(s_sigma).replace('.', 'p')
    tag = "-G" if G_fit else ""

    output_png = os.path.join(
        output_path,
        f"mf-{move_factor_str}-ar-{area_ratio_str}-ss-{s_sigma_str}{tag}{file_tag}.png"
    )
    output_fits = os.path.join(
        output_path,
        f"mf-{move_factor_str}-ar-{area_ratio_str}-ss-{s_sigma_str}{tag}{file_tag}.fits"
    )
    output_params = os.path.join(
        output_path,
        f"mf-{move_factor_str}-ar-{area_ratio_str}-ss-{s_sigma_str}{tag}{file_tag}.csv"
    )

    segmentation_image.save(output_png, 'PNG', quality=1080)
    helper.save_fits_with_header(seg_with_ids, header, output_fits)


    if par_out:

        print("Extracting parameters...")

        segment_ids = np.arange(tree_of_segments.num_leaves(), tree_of_segments.num_vertices())
        label_data = np.full(tree_of_segments.num_vertices(), -1, dtype=np.int32)
        label_data[segment_ids] = np.arange(len(segment_ids))  # label 0 to N-1
        seg_array = hg.reconstruct_leaf_data(tree_of_segments, label_data)

        coords_per_segment = [[] for _ in range(len(segment_ids))]

        for y_ in range(seg_array.shape[0]):
            for x_ in range(seg_array.shape[1]):
                label = seg_array[y_, x_]
                if label >= 0:
                    coords_per_segment[label].append((y_, x_))

        hlr_values = [helper.half_light_radius(image, coords) for coords in coords_per_segment]

        x = x[n_map_segments][tree_of_segments.num_leaves():]
        y = y[n_map_segments][tree_of_segments.num_leaves():]
        ra, dec = helper.sky_coordinates(y, x, header)

        a, b, theta = helper.second_order_moments(tree_of_segments, image.shape[:2], image)
        flux = hg.accumulate_sequential(tree_of_segments, image, hg.Accumulators.sum)

        helper.save_parameters(
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
            hlr_values[::-1],
            file_name=output_params
        )

    if reduce:
        reduced_fits = os.path.join(output_path, f"reduced{file_tag}.fits")
        helper.save_fits_with_header(image - bg_mean, header, reduced_fits)


if __name__ == "__main__":
    main()

