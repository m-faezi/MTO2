from scipy.ndimage import gaussian_filter
import mto2lib.morphological_background as mb
import mto2lib.const_background as cb

def smooth_filter(image, sigma):

    return gaussian_filter(image, sigma)


def estimate_background(image):

    cons_attempt = cb.estimate_background(image)
    bg_map = cons_attempt if cons_attempt is not None else mb.estimate_structural_background(image)

    return bg_map

