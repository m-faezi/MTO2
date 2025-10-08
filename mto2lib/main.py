from mto2lib import io_utils
from mto2lib.parser import make_parser
from mto2lib.validators import validate_crop_coordinates
from datetime import datetime
import os


def setup():

    arguments = make_parser().parse_args()

    arguments.time_stamp = datetime.now().isoformat()
    os.makedirs(arguments.time_stamp, exist_ok=True)

    crop_coords = None

    if arguments.crop:

        crop_coords = validate_crop_coordinates(arguments.crop)

    image, header = io_utils.read_image_data(arguments.file_path, crop_coords)

    if arguments.crop:

        cropped_time_stamp = os.path.join(
            arguments.time_stamp,
            "processed_input.fits"
        )

        io_utils.save_fits_with_header(image, header, cropped_time_stamp)
        print(f"Saved cropped image to: {cropped_time_stamp}")

    return image, header, arguments

