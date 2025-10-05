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

