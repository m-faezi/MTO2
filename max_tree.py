from mto2lib.utils import base_utils
from mto2lib import max_tree_attributes
from mto2lib import statistical_tests
import numpy as np
import higra as hg


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
        self.significant_nodes = None
        self.init_segments = None
        self.corrected_segments = None


    def construct_max_tree(self, image_reduced):

        self.graph, self.tree_structure, self.altitudes = base_utils.image_to_hierarchical_structure(image_reduced)

        return self


    def compute_attributes(self, run, image):

        (self.x, self.y, self.distances, self.distance_to_root_center, self.mean, self.variance, self.area,
        self.parent_area, self.gaussian_intensities, self.volume, self.parent_altitude, self.gamma, self.parent_gamma) \
            = max_tree_attributes.compute_attributes(self.tree_structure, self.altitudes, run, image)

        return self


    def detect_significant_objects(self, dark_frame):

        self.significant_nodes = statistical_tests.attribute_statistical_significance(self, dark_frame)
        self.init_segments = statistical_tests.select_objects(self)

        return self


    def move_up(
            self,
            dark_frame,
            run
    ):

        main_branch_local = statistical_tests.attribute_main_branch(self.tree_structure, self.area)

        closest_object_ancestor = hg.propagate_sequential(
            self.tree_structure,
            np.arange(self.tree_structure.num_vertices()),
            np.logical_and(main_branch_local, np.logical_not(self.init_segments)))

        target_altitudes = self.altitudes.copy()
        object_indexes, = np.nonzero(self.init_segments)
        local_noise = np.sqrt(
            np.maximum(
                np.where(
                    dark_frame.bg_gain != 0,
                    dark_frame.bg_var + self.altitudes[self.tree_structure.parent(object_indexes)],
                    0
                ) / dark_frame.bg_gain,
                0
            )
        )

        target_altitudes[object_indexes] = self.altitudes[object_indexes] + run.arguments.move_factor * local_noise

        target_altitudes = target_altitudes[closest_object_ancestor]

        if not run.arguments.G_fit:

            valid_moves = np.logical_and(
                self.altitudes >= target_altitudes,
                np.logical_and(
                    self.init_segments[closest_object_ancestor],
                    self.area / self.parent_area >= run.arguments.area_ratio
                )
            )

        elif run.arguments.G_fit:

            valid_moves = np.logical_and(
                np.logical_and(
                    self.altitudes >= target_altitudes,
                    np.logical_and(
                        self.init_segments[closest_object_ancestor],
                        self.area / self.parent_area >= run.arguments.area_ratio
                    )
                ),
                self.altitudes / self.area >= self.gaussian_intensities
            )

        parent_closest_object_ancestor = closest_object_ancestor[self.tree_structure.parents()]
        parent_not_valid_moves = np.logical_not(valid_moves[self.tree_structure.parents()])
        self.corrected_segments = np.logical_and(
            valid_moves,
            np.logical_or(
                parent_not_valid_moves,
                parent_closest_object_ancestor != closest_object_ancestor
            )
        )

        return self

