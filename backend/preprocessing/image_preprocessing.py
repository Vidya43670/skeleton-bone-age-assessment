
from pathlib import Path

import cv2
import numpy as np


# Standard image size for CNN input
DEFAULT_IMAGE_SIZE = (224, 224)


def validate_image(image):
    """
    Validate that the input is a valid grayscale image.
    """
    if image is None:
        raise ValueError("Image could not be loaded.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Image is empty.")

    return image


def convert_to_grayscale(image):
    """
    Convert an image to grayscale if it is a color image.
    """
    validate_image(image)

    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def resize_image(image, image_size=DEFAULT_IMAGE_SIZE):
    """
    Resize the image to the required CNN input size.
    """
    validate_image(image)

    return cv2.resize(
        image,
        image_size,
        interpolation=cv2.INTER_AREA
    )


def normalize_image(image):
    """
    Normalize pixel values from the range 0-255 to 0-1.
    """
    validate_image(image)

    image_float = image.astype(np.float32)

    return image_float / 255.0


def baseline_preprocessing(
    image,
    image_size=DEFAULT_IMAGE_SIZE
):
    """
    P0: Baseline preprocessing.

    Steps:
    1. Convert to grayscale
    2. Resize
    3. Normalize
    """
    image = convert_to_grayscale(image)
    image = resize_image(image, image_size)
    image = normalize_image(image)

    return image


def clahe_preprocessing(
    image,
    image_size=DEFAULT_IMAGE_SIZE,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    P1: CLAHE preprocessing.

    CLAHE means Contrast Limited Adaptive
    Histogram Equalization.
    """
    image = convert_to_grayscale(image)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    image = clahe.apply(image)
    image = resize_image(image, image_size)
    image = normalize_image(image)

    return image


def denoising_preprocessing(
    image,
    image_size=DEFAULT_IMAGE_SIZE
):
    """
    P2: Denoising preprocessing.

    Gaussian filtering is used to reduce noise.
    """
    image = convert_to_grayscale(image)

    image = cv2.GaussianBlur(
        image,
        (5, 5),
        sigmaX=0
    )

    image = resize_image(image, image_size)
    image = normalize_image(image)

    return image


def sharpening_preprocessing(
    image,
    image_size=DEFAULT_IMAGE_SIZE
):
    """
    P3: Sharpening preprocessing.

    Unsharp masking is used to enhance edges.
    """
    image = convert_to_grayscale(image)

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=3
    )

    sharpened = cv2.addWeighted(
        image,
        1.5,
        blurred,
        -0.5,
        0
    )

    image = resize_image(sharpened, image_size)
    image = normalize_image(image)

    return image


def apply_preprocessing(
    image,
    preprocessing_id,
    image_size=DEFAULT_IMAGE_SIZE
):
    """
    Apply a preprocessing method using its ID.

    Supported IDs:
    P0 - Baseline
    P1 - CLAHE
    P2 - Denoising
    P3 - Sharpening
    """
    preprocessing_methods = {
        "P0": baseline_preprocessing,
        "P1": clahe_preprocessing,
        "P2": denoising_preprocessing,
        "P3": sharpening_preprocessing
    }

    if preprocessing_id not in preprocessing_methods:
        raise ValueError(
            f"Unknown preprocessing ID: {preprocessing_id}. "
            f"Choose from {list(preprocessing_methods.keys())}."
        )

    method = preprocessing_methods[preprocessing_id]

    return method(image, image_size)