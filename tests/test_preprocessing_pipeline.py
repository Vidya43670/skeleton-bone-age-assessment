
import sys
from pathlib import Path

import cv2
import numpy as np


# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.preprocessing.image_preprocessing import (
    apply_preprocessing
)


def create_test_image():
    """
    Create a synthetic grayscale image.

    This is only for testing the preprocessing code.
    It is not a medical X-ray image.
    """
    image = np.zeros((300, 300), dtype=np.uint8)

    # Add a rectangle
    cv2.rectangle(
        image,
        (50, 50),
        (250, 250),
        180,
        thickness=-1
    )

    # Add a circle
    cv2.circle(
        image,
        (150, 150),
        60,
        255,
        thickness=-1
    )

    # Add artificial noise
    noise = np.random.normal(
        loc=0,
        scale=15,
        size=image.shape
    )

    noisy_image = image.astype(np.float32) + noise

    noisy_image = np.clip(
        noisy_image,
        0,
        255
    ).astype(np.uint8)

    return noisy_image


def test_all_preprocessing_methods():
    """
    Test P0, P1, P2 and P3.
    """
    image = create_test_image()

    preprocessing_ids = [
        "P0",
        "P1",
        "P2",
        "P3"
    ]

    for preprocessing_id in preprocessing_ids:
        processed_image = apply_preprocessing(
            image,
            preprocessing_id
        )

        # Check output dimensions
        assert processed_image.shape == (
            224,
            224
        ), (
            f"Incorrect shape for {preprocessing_id}: "
            f"{processed_image.shape}"
        )

        # Check data type
        assert processed_image.dtype == np.float32, (
            f"Incorrect data type for {preprocessing_id}"
        )

        # Check normalized range
        assert processed_image.min() >= 0.0, (
            f"Minimum value is invalid for {preprocessing_id}"
        )

        assert processed_image.max() <= 1.0, (
            f"Maximum value is invalid for {preprocessing_id}"
        )

        print(
            f"{preprocessing_id}: "
            f"Shape={processed_image.shape}, "
            f"Dtype={processed_image.dtype}, "
            f"Range=({processed_image.min():.4f}, "
            f"{processed_image.max():.4f})"
        )


if __name__ == "__main__":
    test_all_preprocessing_methods()
    print("\nAll preprocessing tests passed.")