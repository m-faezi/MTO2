from mto2lib import parser, io_utils, validators
from mto2lib.parser import make_parser

def setup():
    """Read in a file and parameters; run initialisation functions."""

    arguments = make_parser().parse_args()

    image, header = io_utils.read_image_data(arguments.file_path)

    return image, header, arguments


