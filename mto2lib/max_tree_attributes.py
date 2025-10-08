import mto2lib.utils as uts
import numpy as np
import higra as hg


def compute_attributes(tree_structure, image, image_reduced, altitudes):

    x, y = uts.centroid(tree_structure, image.shape[:2])
    distances = np.sqrt((x[tree_structure.parents()] - x) ** 2 + (y[tree_structure.parents()] - y) ** 2)
    mean, variance = hg.attribute_gaussian_region_weights_model(tree_structure, image_reduced)
    area = hg.attribute_area(tree_structure)
    parent_area = area[tree_structure.parents()]

    gaussian_intensities = uts.compute_gaussian_profile(
        mean,
        variance,
        distances,
        altitudes/area
    )

    volume = hg.attribute_volume(tree_structure, altitudes)
    parent_altitude = altitudes[tree_structure.parents()]
    gamma = hg.attribute_topological_height(tree_structure)
    parent_gamma = gamma[tree_structure.parents()]

    return (x, y, distances, mean, variance, area, parent_area, gaussian_intensities, volume, parent_altitude, gamma,
            parent_gamma)
