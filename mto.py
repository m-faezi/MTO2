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
        type=np.float64,
        default=0.1,
        help='move_factor parameter for isophote correction (default = 0.1)'
    )
    parser.add_argument('--par_out', action='store_true', help='Extract and save parameters, if set')
    parser.add_argument('--reduce', action='store_true', help='Returns background subtracted image')

    args = parser.parse_args()
    data_path = args.file_path
    move_factor = args.move_factor
    par_out = args.par_out
    reduce = args.reduce

    image, header = helper.read_image_data(data_path)
    image = helper.image_value_check(image)
    image_processed = helper.smooth_filter(image)

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
        altitudes
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
        parent_gamma - gamma, gaussian_intensities, move_factor
    )

    tree_of_segments, n_map_segments = hg.simplify_tree(tree_structure, np.logical_not(modified_isophote))

    colors = np.random.randint(0, 255, (tree_of_segments.num_vertices(), 3), dtype=np.uint8)
    colors[tree_of_segments.root(), :] = 0
    seg = hg.reconstruct_leaf_data(tree_of_segments, colors)

    segmentation_image = Image.fromarray(seg.astype(np.uint8))
    segmentation_image = ImageOps.flip(segmentation_image)
    unique_segment_ids = np.arange(tree_of_segments.num_vertices())[::-1]
    seg_with_ids = hg.reconstruct_leaf_data(tree_of_segments, unique_segment_ids)

    move_factor_str = str(move_factor).replace('.', '_')
    output_png = f'MTO-move_factor-{move_factor_str}.png'
    output_fits = f'MTO-move_factor-{move_factor_str}.fits'
    output_params = f'MTO-move_factor-{move_factor_str}.csv'
    reduced_fits = f'MTO-reduced.fits'

    segmentation_image.save(output_png, 'PNG', quality=720)
    helper.save_fits_with_header(seg_with_ids, header, output_fits)

    if par_out:

        print("Extracting parameters...")
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
            file_name=output_params
        )
        print(f"Parameters saved to {output_params}")

    if reduce:
        helper.save_fits_with_header(image - bg_mean, header, reduced_fits)


if __name__ == "__main__":

    main()

