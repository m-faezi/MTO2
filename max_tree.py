from mto2lib.utils import base_utils
from mto2lib import max_tree_attributes


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

