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


class Image:

    def __init__(self):
        self.image = None
        self.smooth_image = None
        self.reduced_image = None
        self.header = None


    def get_image(self, arguments):

        self.image, self.header = mto2.get_image(arguments)

        return self


    def preprocess_image(self, s_sigma):

        self.image = validators.image_value_check(self.image)

        self.smooth_image = preprocessing.smooth_filter(self.image, s_sigma)

        return self


class DarkFrame:

    def __init__(self):
        self.bg_mean = None
        self.bg_var = None
        self.bg_gain = None
        self.bg_map = None

    def estimate_const_bg(self, smooth_image):

        self.bg_mean, self.bg_var, self.bg_gain, self.bg_map = (
            preprocessing.get_constant_background_map(smooth_image)
        )

        return self


    def estimate_morph_bg(self, image, maxtree):

        self.bg_mean, self.bg_var, self.bg_gain, self.bg_map = (
            preprocessing.get_morphological_background_map(image.smooth_image, maxtree)
        )

        return self


    def save_background(self, results_dir, header, arguments):

        bg_output = os.path.join(results_dir, "background_map.fits")
        io_utils.save_fits_with_header(self.bg_map, header, bg_output)

        print(f"Saved {arguments.background_mode} background to: {bg_output}")

        return self

    def create_reduced_image(self, image, results_dir):

        image.reduced_image = image.smooth_image - self.bg_mean
        reduced_output = os.path.join(results_dir, "reduced.fits")
        io_utils.save_fits_with_header(image.reduced_image, image.header, reduced_output)

        print(f"Saved reduced image to: {reduced_output}")

        return self


class MTO2Run:

    def __init__(self):
        self.arguments = None
        self.results_dir = None


    def setup_args(self):

        self.arguments, self.results_dir = mto2.setup_args()

        return self


    def detect_significant_objects(self, dark_frame, tree):

        significant_nodes = statistical_tests.attribute_statistical_significance(
            tree, dark_frame
        )

        tree.init_segments = statistical_tests.select_objects(tree, significant_nodes)

    def refine_isophotes(
        self,
        tree,
        dark_frame,
        run
    ):

        return statistical_tests.move_up(
            tree,
            dark_frame,
            run,
        )

    def create_segmentation(self, tree, image, modified_isophote):

        return segment.get_segmentation_map(
            tree.tree_structure,
            modified_isophote,
            image.header,
            self.arguments
        )

    def extract_parameters(
        self,
        tree_of_segments,
        n_map_segments,
        unique_ids,
        parent_altitude,
        area,
        image
    ):

        if self.arguments.par_out:
            parameter_extraction.extract_parameters(
                image.image,
                image.header,
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
        self.tree_structure = None
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
        self.init_segments = None

    def construct_max_tree(self, image_reduced):

        self.graph, self.tree_structure, self.altitudes = base_utils.image_to_hierarchical_structure(image_reduced)

        return self

    def compute_attributes(self, run, image):

        (self.x, self.y, self.distances, self.distance_to_root_center, self.mean, self.variance, self.area,
        self.parent_area, self.gaussian_intensities, self.volume, self.parent_altitude, self.gamma, self.parent_gamma) \
            = max_tree_attributes.compute_attributes(self.tree_structure, self.altitudes, run, image)

        return self


def execute_run():

        run = MTO2Run()
        image = Image()
        dark_frame = DarkFrame()

    #try:

        run.setup_args()
        image.get_image(run.arguments)

        image.preprocess_image(run.arguments.s_sigma)

        if run.arguments.background_mode == 'const':

            try:

                dark_frame.estimate_const_bg(image.smooth_image)
                dark_frame.create_reduced_image(image, run.results_dir)

                maxtree = MaxTree()
                maxtree.construct_max_tree(image.reduced_image)
                maxtree.compute_attributes(run, image)

            except Exception as e:

                run.arguments.background_mode = 'morph'

                print(f"Note: Background mode switched from \'const\' to '{run.arguments.background_mode}'!")

                maxtree = MaxTree()
                maxtree.construct_max_tree(image.smooth_image)
                maxtree.compute_attributes(run, image)

                dark_frame.estimate_morph_bg(image, maxtree)

        else:

            maxtree = MaxTree()
            maxtree.construct_max_tree(image.smooth_image)
            maxtree.compute_attributes(run, image)

            dark_frame.estimate_morph_bg(image, maxtree)

        io_utils.save_parameters_metadata(
            run.arguments,
            run.results_dir,
        )

        dark_frame.save_background(run.results_dir, image.header, run.arguments)

        dark_frame.create_reduced_image(image, run.results_dir)

        run.detect_significant_objects(dark_frame, maxtree)

        modified_isophote = run.refine_isophotes(maxtree, dark_frame, run)


        tree_of_segments, n_map_segments, unique_ids = run.create_segmentation(
            maxtree, image, modified_isophote
        )

        run.extract_parameters(
            tree_of_segments,
            n_map_segments,
            unique_ids,
            maxtree.parent_altitude,
            maxtree.area,
            image
        )

        io_utils.set_run_status("Completed")

        print("MTO2 run completed successfully!")

#    except Exception as e:
#
#        io_utils.set_run_status("Terminated")
#
#        print(f"MTO2 run terminated with error: {e}")
#
#        sys.exit(1)


if __name__ == "__main__":
    execute_run()

