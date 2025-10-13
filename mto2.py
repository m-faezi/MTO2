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
        self.smooth_image = None
        self.reduced_image = None
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

        self.smooth_image = preprocessing.smooth_filter(self.image, self.arguments.s_sigma)

        return self

    def estimate_const_bg(self):

        self.bg_mean, self.bg_var, self.bg_gain, self.bg_map, self.actual_mode = (
            preprocessing.get_constant_background_map(self.smooth_image)
        )

        return self


    def estimate_morph_bg(self, maxtree):

        self.bg_mean, self.bg_var, self.bg_gain, self.bg_map = (
            preprocessing.get_morphological_background_map(self.smooth_image, maxtree)
        )

        return self


    def save_background(self):

        bg_output = os.path.join(self.results_dir, "background_map.fits")
        io_utils.save_fits_with_header(self.bg_map, self.header, bg_output)

        print(f"Saved {self.actual_mode} background to: {bg_output}")

        return self

    def create_reduced_image(self):

        self.reduced_image = self.smooth_image - self.bg_map
        reduced_output = os.path.join(self.results_dir, "reduced.fits")
        io_utils.save_fits_with_header(self.reduced_image, self.header, reduced_output)

        print(f"Saved reduced image to: {reduced_output}")

        return self


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


class MaxTree:

    def __init__(self):
        self.graph = None
        self.tree = None
        self.altitudes = None
        self.x = None
        self.y = None
        self.distances = None
        self.distance_to_root_center = None
        self.mean = None
        self.variance = None
        self.area = None
        self.parent_area = None
        self.gaussian_intensities = None
        self.volume = None
        self.parent_altitude = None
        self.gamma = None
        self.parent_gamma = None

    def construct_max_tree(self, image_reduced):

        self.graph, self.tree, self.altitudes = base_utils.image_to_hierarchical_structure(image_reduced)

        return self

    def compute_attributes(self, tree_structure, run):

        (self.x, self.y, self.distances, self.distance_to_root_center, self.mean, self.variance, self.area,
        self.parent_area, self.gaussian_intensities, self.volume, self.parent_altitude, self.gamma, self.parent_gamma) \
            = max_tree_attributes.compute_attributes(tree_structure, self.altitudes, run)

        return self


def execute_run():

    run = MTO2Run()

    try:

        run.setup()
        run.preprocess_image()

        if run.arguments.background_mode == 'const':

            try:

                run.estimate_const_bg()

                maxtree = MaxTree()
                maxtree.construct_max_tree(run.smooth_image)
                maxtree.compute_attributes(maxtree.tree, run)

            except Exception as e:

                maxtree = MaxTree()
                maxtree.construct_max_tree(run.smooth_image)
                maxtree.compute_attributes(maxtree.tree, run)

                run.estimate_morph_bg(maxtree)

        else:

            maxtree = MaxTree()
            maxtree.construct_max_tree(run.smooth_image)
            maxtree.compute_attributes(maxtree.tree, run)

            run.estimate_morph_bg(maxtree)

        io_utils.save_parameters_metadata(
            run.arguments,
            run.results_dir,
            actual_background_mode=run.actual_mode
        )

        run.save_background()

        run.create_reduced_image()

        objects = run.detect_significant_objects(maxtree.tree, maxtree.altitudes, maxtree.volume, maxtree.area)

        modified_isophote = run.refine_isophotes(
            maxtree.tree, maxtree.altitudes, maxtree.area, maxtree.parent_area, maxtree.distances, objects,
            maxtree.gaussian_intensities, maxtree.parent_gamma, maxtree.gamma
        )

        tree_of_segments, n_map_segments, unique_ids = run.create_segmentation(
            maxtree.tree, modified_isophote
        )

        run.extract_parameters(
            tree_of_segments,
            n_map_segments,
            unique_ids,
            maxtree.parent_altitude,
            maxtree.area
        )

        io_utils.set_run_status("Completed")

        print("MTO2 run completed successfully!")

    except Exception as e:

        io_utils.set_run_status("Terminated")

        print(f"MTO2 run terminated with error: {e}")

        sys.exit(1)


if __name__ == "__main__":
    execute_run()

