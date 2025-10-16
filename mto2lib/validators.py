import numpy as np
import argparse


def restricted_non_negative(value):

    try:

        value = float(value)

    except ValueError:

        raise argparse.ArgumentTypeError(f"{value} is not a valid float")

    if value < 0:

        raise argparse.ArgumentTypeError(f"{value} not in range [0.0, +∞)")

    return value


def restricted_normal(value):

    try:

        value = float(value)

    except ValueError:

        raise argparse.ArgumentTypeError(f"{value} is not a valid float")

    if value < 0 or value >= 1.0:

        raise argparse.ArgumentTypeError(f"{value} not in range [0.0, 1.0)")

    return value


def validate_crop_coordinates(value, image_shape):

    if not value:

        return None

    try:

        height, width = image_shape
        x1, y1, x2, y2 = value

        x1 = x1 if x1 >= 0 else width+x1
        x2 = x2 if x2 >= 0 else width+x2
        y1 = y1 if y1 >= 0 else height+y1
        y2 = y2 if y2 >= 0 else height+y2

        if x1 >= x2 or y1 >= y2:

            raise ValueError("Coordinates must form a valid rectangle (x1 < x2, y1 < y2)")

        if x1 < 0 or y1 < 0:

            raise ValueError("Coordinates must be non-negative")

        return x1, y1, x2, y2

    except ValueError as e:

        raise argparse.ArgumentTypeError(f"Invalid crop coordinates '{value}': {e}. Format: x1 y1 x2 y2")


def image_value_check(image):

    if np.isnan(image).any():

        min_value = np.nanmin(image)
        image[np.isnan(image)] = min_value

    return np.maximum(image, 0)


def image_value_check_2(image):

    if np.isnan(image).any():

        finite_values = image[np.isfinite(image)]

        if finite_values.size == 0:

            raise ValueError("Image contains no finite values (only NaNs or infinities).")

        min_value = finite_values.min()
        image = np.where(np.isnan(image), min_value, image)

    if np.isinf(image).any():

        finite_values = image[np.isfinite(image)]

        if finite_values.size == 0:

            raise ValueError("Image contains no finite values (only infinities or NaNs).")

        max_value = finite_values.max()
        min_value = finite_values.min()
        image = np.where(image == np.inf, max_value, image)
        image = np.where(image == -np.inf, min_value, image)

    return image

