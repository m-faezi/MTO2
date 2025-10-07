from scipy.ndimage import gaussian_filter
import mto2lib.morphological_background as mb
import mto2lib.const_background as cb


def smooth_filter(image, sigma):

    return gaussian_filter(image, sigma)


def estimate_background(image):

    bg_mean, bg_var, bg_gain, bg_type = cb.estimate_background(image, return_map=False)

    return bg_mean, bg_var, bg_gain


def get_constant_background_map(image):

    result = cb.estimate_background(image, return_map=True)

    if result is not None:

        bg_mean, bg_var, bg_gain, bg_map, bg_type = result

        return bg_mean, bg_var, bg_gain, bg_map, bg_type

    return None, None, None, None, None


def get_morphological_background_map(image):

    m_result = mb.estimate_structural_background(image, return_map=True)

    if m_result is not None:

        m_bg_mean, m_bg_var, m_bg_gain, m_bg_map = m_result

        return m_bg_mean, m_bg_var, m_bg_gain, m_bg_map, 'morph'

    return None, None, None, None, None

