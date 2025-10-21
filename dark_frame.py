from mto2lib import preprocessing
from mto2lib.utils import io_utils
import os


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

        image.reduced_image = image.image - self.bg_mean
        reduced_output = os.path.join(results_dir, "reduced.fits")

        io_utils.save_fits_with_header(image.reduced_image, image.header, reduced_output)

        print(f"Saved reduced image to: {reduced_output}")

        return self

