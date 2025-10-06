from mto2lib import io_utils
from mto2lib.parser import make_parser
from mto2lib.validators import validate_crop_coordinates


def setup():

    arguments = make_parser().parse_args()

    crop_coords = None

    if arguments.crop:
        crop_coords = validate_crop_coordinates(arguments.crop)

    image, header = io_utils.read_image_data(arguments.file_path, crop_coords)

    return image, header, arguments

