
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

CSV_PATH = RAW_DATA_DIR / "boneage-training-dataset.csv"

IMAGE_DIR = RAW_DATA_DIR / "boneage-training-dataset"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

NUM_CLASSES = 4

IMAGE_SIZE = (224, 224)

CLASS_NAMES = {
    0: "0-48 months",
    1: "49-96 months",
    2: "97-144 months",
    3: "145-216 months"
}


# ============================================================
# PREPROCESSING CONFIGURATION
# ============================================================

PREPROCESSING_METHODS = [
    "P0",
    "P1",
    "P2",
    "P3"
]


# ============================================================
# CNN MODELS
# ============================================================

MODEL_NAMES = [
    "resnet18",
    "densenet121",
    "efficientnet_b0"
]


# ============================================================
# TRAINING PARAMETERS
# ============================================================

BATCH_SIZE = 16

NUM_EPOCHS = 10

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

RANDOM_SEED = 42


# ============================================================
# DATA SPLIT CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

RESULTS_DIR = OUTPUT_DIR / "results"

LOG_DIR = OUTPUT_DIR / "logs"


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

if __name__ == "__main__":

    print("Training Configuration")
    print("-" * 40)

    print("Project Root:", PROJECT_ROOT)
    print("CSV Path:", CSV_PATH)
    print("Image Directory:", IMAGE_DIR)

    print("Number of Classes:", NUM_CLASSES)
    print("Image Size:", IMAGE_SIZE)

    print("Batch Size:", BATCH_SIZE)
    print("Number of Epochs:", NUM_EPOCHS)
    print("Learning Rate:", LEARNING_RATE)

    print("Models:", MODEL_NAMES)
    print("Preprocessing:", PREPROCESSING_METHODS)

    print("Device:", DEVICE)