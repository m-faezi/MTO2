import mto2lib.main as mto2
from mto2lib.utils import base_utils, io_utils

from mto2lib import (
    validators,
    preprocessing,
    max_tree_attributes,
    statistical_tests,
    segment,
    parameter_extraction,
)

import os
import sys


def build_hierarchical_structure(image_reduced):

    graph_structure, tree_structure, altitudes = base_utils.image_to_hierarchical_structure(image_reduced)

    return tree_structure, altitudes


class MTO2Run:

    def __init__(self):
        self.image = None
        self.header = None
        self.arguments = None
        self.results_dir = None
        self.actual_mode = None
        self.bg_mean = None
        self.bg_var = None
        self.bg_gain = None
        self.bg_map = None

    def setup(self):

        self.image, self.header, self.arguments, self.results_dir = mto2.setup()

        return self

    def preprocess_image(self):

        self.image = validators.image_value_check(self.image)

        return preprocessing.smooth_filter(self.image, self.arguments.s_sigma)

    def estimate_background(self, image_processed):

        requested_mode = self.arguments.background_mode

        if requested_mode == 'const':

            try:

                self.bg_mean, self.bg_var, self.bg_gain, self.bg_map, self.actual_mode = (
                    preprocessing.get_constant_background_map(image_processed)
                )

            except Exception as e:

                self.actual_mode = 'morph'

                self.bg_mean, self.bg_var, self.bg_gain, self.bg_map = (
                    preprocessing.get_morphological_background_map(image_processed)
                )

        else:

            self.actual_mode = requested_mode

            self.bg_mean, self.bg_var, self.bg_gain, self.bg_map = (
                preprocessing.get_morphological_background_map(image_processed)
            )

        if self.actual_mode != requested_mode:

            print(f"Note: Background mode fell back from '{requested_mode}' to '{self.actual_mode}'!")

        return self

    def save_background(self):

        bg_output = os.path.join(self.results_dir, "background_map.fits")
        io_utils.save_fits_with_header(self.bg_map, self.header, bg_output)

        print(f"Saved {self.actual_mode} background to: {bg_output}")

        return self

    def create_reduced_image(self, image_processed):

        image_reduced = image_processed - self.bg_mean
        reduced_output = os.path.join(self.results_dir, "reduced.fits")
        io_utils.save_fits_with_header(image_reduced, self.header, reduced_output)

        print(f"Saved reduced image to: {reduced_output}")

        return image_reduced

    def compute_attributes(self, tree_structure, image_reduced, altitudes):

        return max_tree_attributes.compute_attributes(
            tree_structure, self.image, image_reduced, altitudes
        )

    def detect_significant_objects(self, tree_structure, altitudes, volume, area):

        significant_nodes = statistical_tests.attribute_statistical_significance(
            tree_structure, altitudes, volume, area, self.bg_var, self.bg_gain
        )

        return statistical_tests.select_objects(tree_structure, significant_nodes)

    def refine_isophotes(
        self,
        tree_structure,
        altitudes,
        area,
        parent_area,
        distances,
        objects,
        gaussian_intensities,
        parent_gamma,
        gamma
    ):

        return statistical_tests.move_up(
            tree_structure,
            altitudes,
            area,
            parent_area,
            distances,
            objects,
            self.bg_var,
            self.bg_gain,
            parent_gamma - gamma,
            gaussian_intensities,
            self.arguments.move_factor,
            self.arguments.G_fit,
            self.arguments.area_ratio,
        )

    def create_segmentation(self, tree_structure, modified_isophote):

        return segment.get_segmentation_map(
            tree_structure,
            modified_isophote,
            self.header,
            self.arguments
        )

    def extract_parameters(
        self,
        tree_of_segments,
        n_map_segments,
        unique_ids,
        parent_altitude,
        area
    ):

        if self.arguments.par_out:
            parameter_extraction.extract_parameters(
                self.image,
                self.header,
                tree_of_segments,
                n_map_segments,
                parent_altitude,
                area,
                unique_ids,
                self.arguments,
            )
        return self


def execute_run():

    run = MTO2Run()

    try:

        run.setup()

        image_processed = run.preprocess_image()


        run.estimate_background(image_processed)

        io_utils.save_parameters_metadata(
            run.arguments,
            run.results_dir,
            actual_background_mode=run.actual_mode
        )

        run.save_background()

        image_reduced = run.create_reduced_image(image_processed)
        tree_structure, altitudes = build_hierarchical_structure(image_reduced)

        (x, y, distances, mean, variance, area, parent_area,
         gaussian_intensities, volume, parent_altitude, gamma,
         parent_gamma) = run.compute_attributes(tree_structure, image_reduced, altitudes)

        objects = run.detect_significant_objects(tree_structure, altitudes, volume, area)

        modified_isophote = run.refine_isophotes(
            tree_structure, altitudes, area, parent_area, distances, objects,
            gaussian_intensities, parent_gamma, gamma
        )

        tree_of_segments, n_map_segments, unique_ids = run.create_segmentation(
            tree_structure, modified_isophote
        )

        run.extract_parameters(
            tree_of_segments,
            n_map_segments,
            unique_ids,
            parent_altitude,
            area
        )

        io_utils.set_run_status("Completed")

        print("MTO2 run completed successfully!")

    except Exception as e:

        io_utils.set_run_status("Terminated")

        print(f"MTO2 run terminated with error: {e}")

        sys.exit(1)


if __name__ == "__main__":
    execute_run()

