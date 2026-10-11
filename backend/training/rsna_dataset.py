
from pathlib import Path

import cv2
import pandas as pd
import torch

from torch.utils.data import Dataset

from backend.preprocessing.image_preprocessing import (
    apply_preprocessing
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 4

IMAGE_SIZE = (224, 224)

AGE_CLASS_RANGES = {
    0: (0, 48),
    1: (49, 96),
    2: (97, 144),
    3: (145, 216)
}


# ============================================================
# AGE TO CLASS
# ============================================================

def age_to_class(age_in_months):
    """
    Convert bone age in months to a class.

    Class 0: 0-48 months
    Class 1: 49-96 months
    Class 2: 97-144 months
    Class 3: 145-216 months
    """

    if pd.isna(age_in_months):
        return None

    age_in_months = float(age_in_months)

    for class_id, (minimum_age, maximum_age) in (
        AGE_CLASS_RANGES.items()
    ):

        if (
            minimum_age
            <= age_in_months
            <= maximum_age
        ):

            return class_id

    return None


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(
    image_path,
    preprocessing_id="P0"
):
    """
    Load, preprocess, and convert an image
    into a PyTorch tensor.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise FileNotFoundError(
            f"Unable to load image: {image_path}"
        )

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    processed_image = apply_preprocessing(
        image,
        preprocessing_id
    )

    processed_image = (
        processed_image.astype("float32")
        / 255.0
    )

    image_tensor = torch.from_numpy(
        processed_image
    )

    # (224, 224) -> (1, 224, 224)
    image_tensor = image_tensor.unsqueeze(0)

    # (1, 224, 224) -> (3, 224, 224)
    image_tensor = image_tensor.repeat(
        3,
        1,
        1
    )

    return image_tensor


# ============================================================
# RSNA DATASET CLASS
# ============================================================

class RSNABoneAgeDataset(Dataset):

    def __init__(
        self,
        csv_path,
        image_dir,
        preprocessing_id="P0",
        transform=None
    ):

        self.csv_path = Path(csv_path)

        self.image_dir = Path(image_dir)

        self.preprocessing_id = (
            preprocessing_id
        )

        self.transform = transform


        # ----------------------------------------------------
        # CHECK PATHS
        # ----------------------------------------------------

        if not self.csv_path.exists():

            raise FileNotFoundError(
                f"CSV file not found: "
                f"{self.csv_path}"
            )

        if not self.image_dir.exists():

            raise FileNotFoundError(
                f"Image directory not found: "
                f"{self.image_dir}"
            )


        # ----------------------------------------------------
        # READ CSV
        # ----------------------------------------------------

        self.data = pd.read_csv(
            self.csv_path
        )


        # ----------------------------------------------------
        # RENAME BONE AGE COLUMN
        # ----------------------------------------------------

        if (
            "boneage" in self.data.columns
            and "bone_age"
            not in self.data.columns
        ):

            self.data = self.data.rename(
                columns={
                    "boneage": "bone_age"
                }
            )


        # ----------------------------------------------------
        # CHECK REQUIRED COLUMNS
        # ----------------------------------------------------

        required_columns = [
            "id",
            "bone_age"
        ]

        for column in required_columns:

            if column not in self.data.columns:

                raise ValueError(
                    f"Required column "
                    f"'{column}' not found"
                )


        # ----------------------------------------------------
        # CONVERT AGE TO CLASS
        # ----------------------------------------------------

        self.data["class_id"] = (
            self.data["bone_age"]
            .apply(age_to_class)
        )


        # ----------------------------------------------------
        # REMOVE INVALID AGES
        # ----------------------------------------------------

        self.data = self.data[
            self.data["class_id"].notna()
        ].copy()


        # ----------------------------------------------------
        # CONVERT CLASS TO INTEGER
        # ----------------------------------------------------

        self.data["class_id"] = (
            self.data["class_id"]
            .astype(int)
        )


        # ----------------------------------------------------
        # RESET INDEX
        # ----------------------------------------------------

        self.data = self.data.reset_index(
            drop=True
        )


        print(
            f"Loaded dataset: "
            f"{self.csv_path.name}"
        )

        print(
            f"Valid samples: "
            f"{len(self.data)}"
        )

        print(
            f"Preprocessing: "
            f"{self.preprocessing_id}"
        )


    # ========================================================
    # DATASET LENGTH
    # ========================================================

    def __len__(self):

        return len(self.data)


    # ========================================================
    # GET DATASET ITEM
    # ========================================================

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_id = str(row["id"])

        image_path = (
            self.image_dir
            / f"{image_id}.png"
        )

        image_tensor = prepare_image(
            image_path=image_path,
            preprocessing_id=(
                self.preprocessing_id
            )
        )

        if self.transform is not None:

            image_tensor = self.transform(
                image_tensor
            )

        label = torch.tensor(
            row["class_id"],
            dtype=torch.long
        )

        return image_tensor, label