
import sys
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.training.rsna_dataset import (
    RSNABoneAgeDataset,
    age_to_class
)


def create_synthetic_dataset(temp_dir):
    """
    Create synthetic images and metadata for testing.

    This is not real RSNA data.
    """

    image_dir = Path(temp_dir) / "images"

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    ages = [
        24,
        72,
        120,
        180
    ]

    for index, age in enumerate(ages, start=1):

        image_id = str(index)

        # Create a synthetic grayscale image
        image = np.random.randint(
            0,
            256,
            size=(300, 300),
            dtype=np.uint8
        )

        image_path = image_dir / f"{image_id}.png"

        cv2.imwrite(
            str(image_path),
            image
        )

        records.append({
            "id": image_id,
            "bone_age": age
        })

    csv_path = Path(temp_dir) / "labels.csv"

    dataframe = pd.DataFrame(records)

    dataframe.to_csv(
        csv_path,
        index=False
    )

    return csv_path, image_dir


def test_dataset_loader():

    print("=" * 60)
    print("RSNA DATASET LOADER TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:

        csv_path, image_dir = create_synthetic_dataset(
            temp_dir
        )

        preprocessing_ids = [
            "P0",
            "P1",
            "P2",
            "P3"
        ]

        for preprocessing_id in preprocessing_ids:

            print("\n" + "-" * 60)
            print(
                f"Testing preprocessing: {preprocessing_id}"
            )
            print("-" * 60)

            dataset = RSNABoneAgeDataset(
                csv_path=csv_path,
                image_dir=image_dir,
                preprocessing_id=preprocessing_id
            )

            # Verify dataset size
            print(
                f"Dataset length: {len(dataset)}"
            )

            assert len(dataset) == 4

            # Load the first sample
            image_tensor, label = dataset[0]

            print(
                f"Image tensor shape: {image_tensor.shape}"
            )

            print(
                f"Image tensor dtype: {image_tensor.dtype}"
            )

            print(
                f"Label: {label.item()}"
            )

            # Verify image tensor
            assert image_tensor.shape == (
                3,
                224,
                224
            )

            assert str(image_tensor.dtype) == (
                "torch.float32"
            )

            # Verify label
            assert label.item() == 0

            # Test DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False
            )

            images, labels = next(
                iter(dataloader)
            )

            print(
                f"Batch image shape: {images.shape}"
            )

            print(
                f"Batch label shape: {labels.shape}"
            )

            assert images.shape == (
                2,
                3,
                224,
                224
            )

            assert labels.shape == (
                2,
            )

            print(
                f"PASS: {preprocessing_id}"
            )

    print("\n" + "=" * 60)
    print("DATASET LOADER TEST COMPLETED")
    print("=" * 60)

    print("\nAll dataset loader tests passed!")


if __name__ == "__main__":
    test_dataset_loader()