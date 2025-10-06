from astropy.io import fits


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

    if 'NAXIS1' in header and 'NAXIS2' in header:
        header['NAXIS1'] = x2 - x1
        header['NAXIS2'] = y2 - y1

        if 'CRPIX1' in header:
            header['CRPIX1'] = header.get('CRPIX1', 1) - x1
        if 'CRPIX2' in header:
            header['CRPIX2'] = header.get('CRPIX2', 1) - y1

    return cropped_image, header


def save_fits_with_header(data, header, output_path):

    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(output_path, overwrite=True)

    return None


def get_output_name(arguments):

    move_factor_str = str(arguments.move_factor).replace('.', 'p')
    area_ratio_str = str(arguments.area_ratio).replace('.', 'p')
    s_sigma_str = str(arguments.s_sigma).replace('.', 'p')

    base_name = f"mf-{move_factor_str}-ar-{area_ratio_str}-ss-{s_sigma_str}"

    if arguments.G_fit:
        base_name += "-G"

    if arguments.file_tag:
        base_name += f"-{arguments.file_tag}"

    if arguments.crop:
        crop_str = arguments.crop.replace(',', '_').replace(' ', '')
        base_name += f"-crop_{crop_str}"

    return base_name

