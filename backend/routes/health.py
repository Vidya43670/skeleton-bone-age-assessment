"""
Health Check and System Diagnostics API Endpoint.
Exposes GET /api/health with hardware device status, database status,
dataset presence flag, and system configuration.
"""

from flask import Blueprint, jsonify
import torch
import platform
import os
import backend.config as config
from backend.database.db import SessionLocal
from backend.database.models import Experiment

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    Comprehensive system health check.
    Returns hardware specs, database connectivity, and dataset presence.
    """
    # Test database connectivity
    db_connected = False
    experiment_count = 0
    try:
        db = SessionLocal()
        experiment_count = db.query(Experiment).count()
        db_connected = True
        db.close()
    except Exception as e:
        db_connected = False

    # Check dataset presence in data/raw
    dataset_present = False
    csv_exists = False
    images_found = 0

    if config.DATA_RAW_DIR.exists():
        # Check for CSV file
        csv_candidates = list(config.DATA_RAW_DIR.glob("*.csv"))
        csv_exists = len(csv_candidates) > 0

        # Check for image files in raw dir or subdirectories
        image_extensions = ("*.png", "*.jpg", "*.jpeg")
        for ext in image_extensions:
            images_found += len(list(config.DATA_RAW_DIR.glob(ext)))
            images_found += len(list(config.DATA_RAW_DIR.glob(f"**/{ext}")))

        dataset_present = csv_exists or (images_found > 0)

    response = {
        "status": "healthy",
        "timestamp": platform.uname().node,
        "disclaimer": config.DISCLAIMER_TEXT,
        "hardware": {
            "device": config.DEVICE,
            "device_name": config.DEVICE_NAME,
            "cuda_available": torch.cuda.is_available(),
            "pytorch_version": torch.__version__,
            "python_version": platform.python_version(),
            "os": platform.platform()
        },
        "database": {
            "status": "connected" if db_connected else "error",
            "sqlite_path": str(config.DATABASE_PATH),
            "seeded_experiments": experiment_count
        },
        "dataset": {
            "present": dataset_present,
            "csv_found": csv_exists,
            "images_detected": images_found,
            "raw_directory": str(config.DATA_RAW_DIR)
        },
        "research_matrix": {
            "preprocessing_methods_count": len(config.PREPROCESSING_METHODS),
            "cnn_architectures_count": len(config.CNN_ARCHITECTURES),
            "total_combinations": len(config.PREPROCESSING_METHODS) * len(config.CNN_ARCHITECTURES),
            "age_classes_count": len(config.AGE_CLASSES)
        }
    }

    return jsonify(response), 200
