from astropy.io import fits
import json
import os
import pandas as pd


def save_run_metadata(run):

    metadata = {
        "software": "MTO2",
        "version": "1.0.0",
        "time_stamp": run.arguments.time_stamp,
        "file_name": os.path.splitext(os.path.basename(run.arguments.file_path))[0],
        "arguments": {
            "background_mode": run.arguments.background_mode,
            "move_factor": run.arguments.move_factor,
            "area_ratio": run.arguments.area_ratio,
            "s_sigma": run.arguments.s_sigma,
            "G_fit": run.arguments.G_fit,
            "crop": run.arguments.crop
        }
    }

    metadata_file = os.path.join(run.results_dir, "run_metadata.json")

    with open(metadata_file, 'w') as f:

        json.dump(metadata, f, indent=2)

    print(f"Saved argument metadata to: {metadata_file}")

    save_run_record(run)

    return metadata_file


def save_run_record(run):

    run_csv_path = os.path.join("./results", "your_runs.csv")

    run_record = {
        "run_id": run.arguments.time_stamp,
        "file_name": os.path.splitext(os.path.basename(run.arguments.file_path))[0],
        "background_mode": run.arguments.background_mode,
        "move_factor": run.arguments.move_factor,
        "area_ratio": run.arguments.area_ratio,
        "s_sigma": run.arguments.s_sigma,
        "G_fit": run.arguments.G_fit,
        "crop": run.arguments.crop,
        "status": run.status,
    }

    if os.path.exists(run_csv_path):

        try:
            existing_df = pd.read_csv(run_csv_path)

            if run.arguments.time_stamp in existing_df['run_id'].values:

                existing_df.loc[existing_df['run_id'] == run.arguments.time_stamp, 'status'] = run.status
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

    if run.status == "Running":

        print(f"Run record created in: {run_csv_path}")

    else:

        print(f"Run marked as {run.status} in: {run_csv_path}")


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


