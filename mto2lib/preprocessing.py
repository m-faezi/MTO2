from scipy.ndimage import gaussian_filter
from mto2lib.reduction import morphological_background as mb
from mto2lib.reduction import const_background as cb


def smooth_filter(image, sigma):

    return gaussian_filter(image, sigma)


def get_constant_background_map(image):

    return cb.estimate_background(image)


def get_morphological_background_map(image, maxtree):

    return mb.estimate_structural_background(image, maxtree)


