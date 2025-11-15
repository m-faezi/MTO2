from mto2lib import validators, preprocessing
import mto2lib.main as mto2
from mto2lib.utils import io_utils

class Image:

    def __init__(self):
        self.image = None
        self.smooth_image = None
        self.reduced_image = None
        self.smooth_reduced_image = None
        self.header = None

    def get_image(self, run):

        self.image, self.header = mto2.get_image(run.arguments, run.results_dir)
        io_utils.save_run_metadata(run)

        return self

    def preprocess_image(self, s_sigma):

        self.image = validators.image_value_check(self.image)
        self.smooth_image = preprocessing.smooth_filter(self.image, s_sigma)

        return self

