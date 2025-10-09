from astropy.io import fits
import json
import os
import pandas as pd
import atexit
import signal
import sys


_run_status = "Running"


def set_run_status(status):

    global _run_status
    _run_status = status


def signal_handler(signum, frame):

    set_run_status("Terminated")
    sys.exit(1)


def register_signal_handlers():

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def save_parameters_metadata(arguments, results_dir, actual_background_mode=None):

    background_mode_used = actual_background_mode if actual_background_mode else arguments.background_mode

    metadata = {
        "software": "MTO2",
        "version": "1.0.0",
        "time_stamp": arguments.time_stamp,
        "file name": os.path.splitext(os.path.basename(arguments.file_path))[0],
        "arguments": {
            "background_mode_requested": arguments.background_mode,
            "background_mode_used": background_mode_used,
            "move_factor": arguments.move_factor,
            "area_ratio": arguments.area_ratio,
            "s_sigma": arguments.s_sigma,
            "G_fit": arguments.G_fit,
            "crop": arguments.crop if arguments.crop else None
        }
    }

    metadata_file = os.path.join(results_dir, "run_metadata.json")

    with open(metadata_file, 'w') as f:

        json.dump(metadata, f, indent=2)

    print(f"Saved argument metadata to: {metadata_file}")

    save_run_record(arguments, background_mode_used, "Running")

    register_signal_handlers()

    atexit.register(finalize_run_record, arguments, background_mode_used)

    return metadata_file


def finalize_run_record(arguments, background_mode_used):

    set_run_status("Completed")
    save_run_record(arguments, background_mode_used, _run_status)


def save_run_record(arguments, background_mode_used, status="Running"):

    run_csv_path = os.path.join("./results", "run_tracker.csv")

    run_record = {
        "run_id": arguments.time_stamp,
        "file_name": os.path.splitext(os.path.basename(arguments.file_path))[0],
        "background_mode_requested": arguments.background_mode,
        "background_mode_used": background_mode_used,
        "move_factor": arguments.move_factor,
        "area_ratio": arguments.area_ratio,
        "s_sigma": arguments.s_sigma,
        "G_fit": arguments.G_fit,
        "crop": str(arguments.crop) if arguments.crop else "None",
        "status": status,
    }

    if os.path.exists(run_csv_path):

        try:
            existing_df = pd.read_csv(run_csv_path)

            if arguments.time_stamp in existing_df['run_id'].values:

                existing_df.loc[
                    existing_df['run_id'] == arguments.time_stamp, list(run_record.keys())] = list(
                    run_record.values())
                updated_df = existing_df

            else:

                new_df = pd.DataFrame([run_record])
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)

        except Exception as e:

            print(f"Warning: Could not read existing run CSV. Creating new one. Error: {e}")

            updated_df = pd.DataFrame([run_record])

    else:

        updated_df = pd.DataFrame([run_record])

    os.makedirs(os.path.dirname(run_csv_path), exist_ok=True)

    updated_df.to_csv(run_csv_path, index=False)

    if status == "Running":

        print(f"Run record created in: {run_csv_path}")

    else:

        print(f"Run marked as {status} in: {run_csv_path}")


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


def save_fits_with_header(data, header, time_stamp):

    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(time_stamp, overwrite=True)

    return None


