from mto2lib import io_utils
import os


def get_image(run):

    image, header = io_utils.read_image_data(run)

    return image, header

