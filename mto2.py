import mto2lib.main as mto2
from mto2lib import (validators, preprocessing, utils, max_tree_attributes, statistical_tests, segment,
                     parameter_extraction, io_utils)
import os

image, header, arguments = mto2.setup()
os.makedirs(arguments.output_path, exist_ok=True)

image = validators.image_value_check(image)
image_processed = preprocessing.smooth_filter(image, arguments.s_sigma)

if arguments.get_cons_background:

    bg_mean, bg_var, bg_gain, bg_map = preprocessing.get_constant_background_map(image_processed)

    if bg_map is not None:

        cons_bg_output = os.path.join(
            arguments.output_path,
            f"background_constant{'-' + arguments.file_tag if arguments.file_tag else ''}.fits"
        )

        io_utils.save_fits_with_header(bg_map, header, cons_bg_output)

        print(f"Saved constant background to: {cons_bg_output}")

if arguments.get_morph_background:

    morph_bg_mean, morph_bg_var, morph_bg_gain, morph_bg_map = preprocessing.get_morphological_background_map(
        image_processed
    )

    if morph_bg_map is not None:

        morph_bg_output = os.path.join(
            arguments.output_path,
            f"background_morphological{'-' + arguments.file_tag if arguments.file_tag else ''}.fits"
        )

        io_utils.save_fits_with_header(morph_bg_map, header, morph_bg_output)

        print(f"Saved morphological background to: {morph_bg_output}")

bg_mean, bg_var, bg_gain = preprocessing.estimate_background(image_processed)
image_reduced = image_processed - bg_mean

graph_structure, tree_structure, altitudes = utils.image_to_hierarchical_structure(image_reduced)

x, y, distances, mean, variance, area, parent_area, gaussian_intensities, volume, parent_altitude, gamma, parent_gamma \
    = max_tree_attributes.compute_attributes(tree_structure, image, image_reduced, altitudes)

significant_nodes = statistical_tests.attribute_statistical_significance(
    tree_structure, altitudes, volume, area, bg_var, bg_gain
)
objects = statistical_tests.select_objects(tree_structure, significant_nodes)

modified_isophote = statistical_tests.move_up(
    tree_structure, altitudes, area, parent_area, distances, objects, bg_var, bg_gain,
    parent_gamma - gamma, gaussian_intensities, arguments.move_factor, arguments.G_fit, arguments.area_ratio
)

name_string = io_utils.get_output_name(arguments)

tree_of_segments, n_map_segments, unique_ids = segment.get_segmentation_map(
    tree_structure,
    modified_isophote,
    header,
    arguments,
    name_string
)

if arguments.par_out:

    parameter_extraction.extract_parameters(
        image,
        header,
        tree_of_segments,
        n_map_segments,
        parent_altitude,
        area,
        unique_ids,
        arguments,
        name_string
    )

if arguments.reduce:

    reduced_fits = os.path.join(
        arguments.output_path,
        f"reduced{'-' + arguments.file_tag if arguments.file_tag else ''}.fits"
    )

    io_utils.save_fits_with_header(image - bg_mean, header, reduced_fits)

