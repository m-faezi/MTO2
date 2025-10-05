from mto2lib import io_utils
from mto2lib.parser import make_parser

def setup():

    arguments = make_parser().parse_args()
    image, header = io_utils.read_image_data(arguments.file_path)

    return image, header, arguments

