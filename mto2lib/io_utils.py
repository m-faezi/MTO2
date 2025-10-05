from astropy.io import fits


def read_image_data(file_path):

    with fits.open(file_path) as hdu_list:

        image_hdu = next((hdu for hdu in hdu_list if hdu.data is not None), None)

        if image_hdu is None:
            raise ValueError("No valid image data found in the FITS file.")

        image = image_hdu.data
        header = image_hdu.header.copy()

    return image, header


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

    return base_name

