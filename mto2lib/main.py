from mto2lib import io_utils
from mto2lib.validators import validate_crop_coordinates
import os


def get_image(arguments, results_dir):

    image, header = io_utils.read_image_data(arguments.file_path)

    if arguments.crop:

        arguments.crop = validate_crop_coordinates(arguments.crop, image.shape)

        image, header = io_utils.read_image_data(arguments.file_path, arguments.crop)

        cropped_file = os.path.join(results_dir, "cropped_input.fits")

        io_utils.save_fits_with_header(image, header, cropped_file)

        print(f"Saved cropped image to: {cropped_file}")

    return image, header

