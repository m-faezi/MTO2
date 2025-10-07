from mto2lib import io_utils
from mto2lib.parser import make_parser
from mto2lib.validators import validate_crop_coordinates
import os


def setup():

    arguments = make_parser().parse_args()

    os.makedirs(arguments.output_path, exist_ok=True)

    crop_coords = None

    if arguments.crop:
        crop_coords = validate_crop_coordinates(arguments.crop)

    image, header = io_utils.read_image_data(arguments.file_path, crop_coords)

    if arguments.crop:

        base_name = os.path.splitext(os.path.basename(arguments.file_path))[0]
        crop_str = f"{arguments.crop[0]}_{arguments.crop[1]}_{arguments.crop[2]}_{arguments.crop[3]}"
        cropped_output_path = os.path.join(
            arguments.output_path,
            f"{base_name}_crop_{crop_str}.fits"
        )

        io_utils.save_fits_with_header(image, header, cropped_output_path)
        print(f"Saved cropped image to: {cropped_output_path}")

    return image, header, arguments

