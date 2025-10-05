import argparse
import mto2lib.validators as validators


def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Path to the image file')
    parser.add_argument(
        '--move_factor',
        type=validators.restricted_non_negative,
        default=0,
        help='move_factor parameter for isophote correction (default = 0)'
    )
    parser.add_argument(
        '--area_ratio',
        type=validators.restricted_normal,
        default=0.90,
        help='area_ratio parameter for deblending correction (default = .90)'
    )
    parser.add_argument(
        '--par_out',
        action='store_true',
        help='Extract and save parameters, if set'
    )
    parser.add_argument(
        '--s_sigma',
        type=validators.restricted_non_negative,
        default=2,
        help='Standard deviation for smoothing Gaussian kernel'
    )
    parser.add_argument(
        '--G_fit',
        action='store_true',
        help='Applies morphological Gaussian filter'
    )
    parser.add_argument(
        '--reduce',
        action='store_true',
        help='Returns background subtracted image'
    )
    parser.add_argument(
        '--file_tag',
        type=str,
        default='',
        help='Optional string to append to saved file names'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='.',
        help='Directory to save output files (default: current directory)'
    )

    return parser

