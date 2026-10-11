"""
Global Configuration for Skeleton Bone Age Assessment System.
Contains paths, hardware device detection, class mapping definitions,
preprocessing parameters, and CNN model specifications.
"""

from pathlib import Path
import os
import torch

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
RESULTS_DIR = BASE_DIR / "results"
DATABASE_PATH = BACKEND_DIR / "database" / "bone_age.db"

# Ensure runtime directories exist
for p in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_METADATA_DIR, MODELS_DIR, EXPERIMENTS_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Hardware execution detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# Reproducibility Seed
RANDOM_SEED = 42

# Academic 4-Class Categorization Ranges (in months)
# Note: Initial research-defined categories (not clinically validated)
AGE_CLASSES = [
    {
        "class_id": 0,
        "name": "Infancy & Early Childhood",
        "min_months": 0,
        "max_months": 48,
        "label": "0–48 months (0–4 yrs)"
    },
    {
        "class_id": 1,
        "name": "Middle Childhood",
        "min_months": 49,
        "max_months": 96,
        "label": "49–96 months (4.1–8 yrs)"
    },
    {
        "class_id": 2,
        "name": "Late Childhood / Early Puberty",
        "min_months": 97,
        "max_months": 144,
        "label": "97–144 months (8.1–12 yrs)"
    },
    {
        "class_id": 3,
        "name": "Adolescence",
        "min_months": 145,
        "max_months": 216,
        "label": "145–216 months (12.1–18 yrs)"
    }
]

NUM_CLASSES = len(AGE_CLASSES)


def map_age_to_class(age_in_months: float):
    """
    Convert bone age in months into one of four research-defined classes.

    Class 0: 0-48 months
    Class 1: 49-96 months
    Class 2: 97-144 months
    Class 3: 145-216 months

    Returns:
        Integer class from 0 to 3.
        Returns None if age is outside the defined ranges.
    """

    if age_in_months is None:
        return None

    age_in_months = float(age_in_months)

    if 0 <= age_in_months <= 48:
        return 0
    elif 49 <= age_in_months <= 96:
        return 1
    elif 97 <= age_in_months <= 144:
        return 2
    elif 145 <= age_in_months <= 216:
        return 3
    else:
        return None
    """
    Configurable function mapping continuous bone age (months) to academic discrete classes.
    Returns class index (0, 1, 2, or 3). Clamps if slightly out of nominal bounds.
    """
    if age_in_months <= 48:
        return 0
    elif age_in_months <= 96:
        return 1
    elif age_in_months <= 144:
        return 2
    else:
        return 3


# Preprocessing Methods Definition (P0 - P3)
PREPROCESSING_METHODS = {
    "P0_baseline": {
        "id": "P0_baseline",
        "name": "Baseline",
        "code": "P0",
        "description": "Grayscale standard normalization, resizing to target dimension, and training-only augmentation.",
        "params": {
            "target_size": [224, 224],
            "normalization": "ImageNet"
        }
    },
    "P1_clahe": {
        "id": "P1_clahe",
        "name": "CLAHE",
        "code": "P1",
        "description": "Contrast Limited Adaptive Histogram Equalization to enhance local bone trabeculae and epiphyseal contrast.",
        "params": {
            "clip_limit": 2.0,
            "tile_grid_size": [8, 8]
        }
    },
    "P2_denoising": {
        "id": "P2_denoising",
        "name": "Denoising",
        "code": "P2",
        "description": "Edge-preserving or Gaussian filtering to attenuate high-frequency radiologic detector noise.",
        "params": {
            "filter_type": "gaussian",  # or "median"
            "kernel_size": 3,
            "sigma": 0.8
        }
    },
    "P3_sharpening": {
        "id": "P3_sharpening",
        "name": "Sharpening",
        "code": "P3",
        "description": "Unsharp masking filter to enhance fine trabecular bone contours and physis-metaphysis boundaries.",
        "params": {
            "method": "unsharp_mask",
            "amount": 1.5,
            "radius": 1.0,
            "threshold": 0
        }
    }
}

# CNN Architecture Specifications
CNN_ARCHITECTURES = {
    "resnet18": {
        "id": "resnet18",
        "name": "ResNet-18",
        "type": "Residual Network",
        "num_classes": NUM_CLASSES,
        "weights": "DEFAULT",
        "description": "Deep residual network with 18 layers using skip connections to avoid gradient degradation."
    },
    "densenet121": {
        "id": "densenet121",
        "name": "DenseNet-121",
        "type": "Densely Connected Convolutional Network",
        "num_classes": NUM_CLASSES,
        "weights": "DEFAULT",
        "description": "Densely connected convolutional network connecting all layers directly for maximum feature reuse."
    },
    "efficientnet_b0": {
        "id": "efficientnet_b0",
        "name": "EfficientNet-B0",
        "type": "Compound Scaled Network",
        "num_classes": NUM_CLASSES,
        "weights": "DEFAULT",
        "description": "Compound scaling architecture balancing network depth, width, and resolution with MBConv blocks."
    }
}

# Default Training Hyperparameters
DEFAULT_TRAINING_CONFIG = {
    "image_size": 224,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 0.0005,
    "optimizer": "Adam",
    "weight_decay": 1e-4,
    "early_stopping_patience": 3,
    "smoke_test_samples": 20
}

# Medical Disclaimer notice
DISCLAIMER_TEXT = (
    "RESEARCH PROTOTYPE DISCLAIMER: This system is developed strictly for academic research "
    "and scientific experimentation under the RSNA Pediatric Bone Age Challenge framework. "
    "It is NOT a medical device, is NOT clinically validated, and MUST NOT be used for clinical "
    "diagnosis, treatment decisions, or medical practice."
)
