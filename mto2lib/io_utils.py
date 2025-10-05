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

    return f"mf-{str(arguments.move_factor).replace('.', 'p')}-ar-{str(arguments.area_ratio).replace('.', 'p')}-ss-{str(arguments.s_sigma).replace('.', 'p')}{'-G' if arguments.G_fit else ''}{arguments.file_tag}"
