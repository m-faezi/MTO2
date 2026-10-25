from mto2lib.utils import base_utils as uts
import numpy as np
import higra as hg


def _as(a, dtype):
    return np.asarray(a).astype(dtype, copy=False)


def compute_attributes(tree_structure, altitudes, run, image):

    parents = tree_structure.parents()

    x, y = uts.centroid(tree_structure, image.image.shape[:2])
    x, y = _as(x, np.float32), _as(y, np.float32)

    mean, variance = hg.attribute_gaussian_region_weights_model(tree_structure, image.image)
    mean, variance = _as(mean, np.float32), _as(variance, np.float32)

    area = hg.attribute_area(tree_structure)
    convexness = _as(hg.attribute_compactness(tree_structure, area=area, normalize=True), np.float32)
    area = _as(area, np.int32)
    parent_area = area[parents]

    parent_altitude = altitudes[parents]
    volume = _as(hg.attribute_volume(tree_structure, altitudes), np.float32)

    gamma = _as(hg.attribute_topological_height(tree_structure), np.int32)
    parent_gamma = gamma[parents]

    return (x, y, mean, variance, area, parent_area, volume,
            parent_altitude, gamma, parent_gamma, convexness)

