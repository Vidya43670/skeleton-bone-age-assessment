
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "boneage-training-dataset.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

RANDOM_SEED = 42

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15


AGE_CLASS_RANGES = {
    0: (0, 48),
    1: (49, 96),
    2: (97, 144),
    3: (145, 216)
}


# ============================================================
# AGE TO CLASS
# ============================================================

def age_to_class(bone_age):
    """
    Convert bone age into one of four
    research-defined classes.

    Returns None for ages outside 0-216 months.
    """

    if pd.isna(bone_age):
        return None

    bone_age = float(bone_age)

    for class_id, (minimum, maximum) in AGE_CLASS_RANGES.items():

        if minimum <= bone_age <= maximum:
            return class_id

    return None


# ============================================================
# CREATE DATA SPLITS
# ============================================================

def main():

    print("Loading dataset...")

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    data = pd.read_csv(CSV_PATH)

    # Support the original RSNA column name
    if (
        "boneage" in data.columns
        and "bone_age" not in data.columns
    ):

        data.rename(
            columns={
                "boneage": "bone_age"
            },
            inplace=True
        )

    required_columns = [
        "id",
        "bone_age"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    # Create class labels
    data["class_id"] = data[
        "bone_age"
    ].apply(age_to_class)

    # Remove records outside defined ranges
    data = data[
        data["class_id"].notna()
    ].copy()

    data["class_id"] = data[
        "class_id"
    ].astype(int)

    data.reset_index(
        drop=True,
        inplace=True
    )

    print("Valid records:", len(data))

    print("\nOverall class distribution:")
    print(data["class_id"].value_counts().sort_index())

    # --------------------------------------------------------
    # TRAIN AND TEMPORARY SPLIT
    # --------------------------------------------------------

    train_data, temporary_data = train_test_split(
        data,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=data["class_id"]
    )

    # --------------------------------------------------------
    # VALIDATION AND TEST SPLIT
    # --------------------------------------------------------

    relative_test_ratio = (
        TEST_RATIO
        / (VALIDATION_RATIO + TEST_RATIO)
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        stratify=temporary_data["class_id"]
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_path = SPLIT_DIR / "train.csv"

    validation_path = SPLIT_DIR / "validation.csv"

    test_path = SPLIT_DIR / "test.csv"

    # Save split files
    train_data.to_csv(
        train_path,
        index=False
    )

    validation_data.to_csv(
        validation_path,
        index=False
    )

    test_data.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY SPLIT INFORMATION
    # --------------------------------------------------------

    print("\nSplit sizes:")
    print("Training:", len(train_data))
    print("Validation:", len(validation_data))
    print("Testing:", len(test_data))

    print("\nTraining class distribution:")
    print(train_data["class_id"].value_counts().sort_index())

    print("\nValidation class distribution:")
    print(
        validation_data["class_id"]
        .value_counts()
        .sort_index()
    )

    print("\nTesting class distribution:")
    print(
        test_data["class_id"]
        .value_counts()
        .sort_index()
    )

    print("\nSaved files:")
    print(train_path)
    print(validation_path)
    print(test_path)

    print("\nDataset splitting completed successfully.")


if __name__ == "__main__":

    main()