from mto2lib.utils import base_utils as uts
import numpy as np
import higra as hg


def compute_attributes(tree_structure, altitudes, run, image):

    x, y = uts.centroid(tree_structure, image.image.shape[:2])
    distances = np.sqrt((x[tree_structure.parents()] - x) ** 2 + (y[tree_structure.parents()] - y) ** 2)
    distance_to_root_center = np.sqrt((x[tree_structure.root()] - x) ** 2 + (y[tree_structure.root()] - y) ** 2)
    mean, variance = hg.attribute_gaussian_region_weights_model(tree_structure, image.image)
    area = hg.attribute_area(tree_structure)
    parent_area = area[tree_structure.parents()]
    parent_altitude = altitudes[tree_structure.parents()]

    if run.arguments.G_fit or run.arguments.background_mode == 'morph':

        gaussian_intensities = uts.compute_gaussian_profile(
            variance,
            distances,
            altitudes/area
        )

    else:

        gaussian_intensities = None

    volume = hg.attribute_volume(tree_structure, altitudes)
    gamma = hg.attribute_topological_height(tree_structure)
    parent_gamma = gamma[tree_structure.parents()]

    return (x, y, distances, distance_to_root_center, mean, variance, area, parent_area, gaussian_intensities, volume,
            parent_altitude, gamma, parent_gamma)

