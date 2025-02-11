import higra as hg
import numpy as np

import background
import helper
from PIL import Image, ImageOps
from matplotlib import pyplot as plt


data_path = 'data/crop_ngc4307_g.fits'
image, header = helper.read_image_data(data_path, 0, 5, 0, 5)
image = helper.image_value_check(image)
image = helper.smooth_filter(image)

_, _, _ = background.estimate_background(image)
