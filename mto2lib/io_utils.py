from astropy.io import fits
import json
from datetime import datetime
import os


def save_parameters_metadata(arguments, output_path, actual_background_mode=None):
    """Save processing parameters as JSON metadata"""

    # Use actual background mode if provided (for fallback cases), otherwise use requested
    background_mode_used = actual_background_mode if actual_background_mode else arguments.background_mode

    metadata = {
        "software": "MTO2",
        "version": "1.0.0",
        "processing_date": datetime.now().isoformat(),
        "parameters": {
            "background_mode_requested": arguments.background_mode,
            "background_mode_used": background_mode_used,
            "move_factor": arguments.move_factor,
            "area_ratio": arguments.area_ratio,
            "s_sigma": arguments.s_sigma,
            "G_fit": arguments.G_fit,
            "file_tag": arguments.file_tag if arguments.file_tag else "",
            "output_path": arguments.output_path,
            "crop": arguments.crop if arguments.crop else None
        }
    }

    metadata_file = os.path.join(output_path, "metadata.json")

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved parameters metadata to: {metadata_file}")
    return metadata_file


def read_image_data(file_path, crop_coords=None):
    with fits.open(file_path) as hdu_list:
        image_hdu = next((hdu for hdu in hdu_list if hdu.data is not None), None)

        if image_hdu is None:
            raise ValueError("No valid image data found in the FITS file.")

        image = image_hdu.data
        header = image_hdu.header.copy()

    if crop_coords:
        image, header = apply_crop(image, header, crop_coords)

    return image, header


def apply_crop(image, header, crop_coords):
    x1, y1, x2, y2 = crop_coords

    height, width = image.shape

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    cropped_image = image[y1:y2, x1:x2]

    if 'NAXIS1' in header:
        header['NAXIS1'] = x2 - x1
    if 'NAXIS2' in header:
        header['NAXIS2'] = y2 - y1

    if 'CRPIX1' in header:
        header['CRPIX1'] = max(1, header.get('CRPIX1', 1) - x1)
    if 'CRPIX2' in header:
        header['CRPIX2'] = max(1, header.get('CRPIX2', 1) - y1)

    return cropped_image, header


def save_fits_with_header(data, header, output_path):
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(output_path, overwrite=True)
    return None


def get_output_name(arguments):
    """Generate minimal base name"""
    base_name = "mto2"

    if arguments.file_tag:
        base_name += f"_{arguments.file_tag}"

    return base_name

