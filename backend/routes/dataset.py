"""
Dataset Management & Validation API Endpoints.
GET /api/dataset/summary
POST /api/dataset/validate
GET /api/dataset/samples
GET /api/dataset/image/<img_id>
GET /api/dataset/splits
"""

from flask import Blueprint, jsonify, request, send_file, abort
from pathlib import Path
import os
import json
import backend.config as config
from backend.database.db import SessionLocal
from backend.database.models import DatasetSummary
import backend.services.dataset_service as dataset_service

dataset_bp = Blueprint("dataset", __name__)


@dataset_bp.route("/dataset/summary", methods=["GET"])
def get_dataset_summary():
    """Retrieve the latest dataset analysis summary and validation report."""
    db = SessionLocal()
    try:
        latest = db.query(DatasetSummary).order_by(DatasetSummary.id.desc()).first()
        if latest:
            return jsonify({"status": "success", "data": latest.to_dict()}), 200

        # If no DB record exists yet, perform a live passive scan using the service
        csv_path, images_dir, image_paths = dataset_service.find_rsna_paths()

        return jsonify({
            "status": "success",
            "data": {
                "is_detected": bool(csv_path and image_paths),
                "csv_found": bool(csv_path),
                "csv_path": str(csv_path) if csv_path else None,
                "images_dir": str(images_dir) if images_dir else str(config.DATA_RAW_DIR),
                "total_images_found": len(image_paths),
                "total_csv_records": 0,
                "valid_paired_records": 0,
                "missing_images_count": 0,
                "invalid_records_count": 0,
                "duplicate_records_count": 0,
                "out_of_range_count": 0,
                "corrupted_images_count": 0,
                "unmatched_csv_records": None,
                "unmatched_image_files": None,
                "age_stats": None,
                "class_distribution": None,
                "class_imbalance_ratio": None,
                "gender_distribution": None,
                "split_stats": None,
                "status_message": (
                    "Dataset pending ingestion. Please place 'boneage-training-dataset.csv' "
                    f"and image files in '{config.DATA_RAW_DIR}'."
                ) if not csv_path else "Files detected. Run validation to compute class distributions and splits.",
                "validated_at": None,
                "age_classes_spec": config.AGE_CLASSES
            }
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        db.close()


@dataset_bp.route("/dataset/validate", methods=["POST"])
def validate_dataset():
    """
    Triggers full or small-sample dataset validation.
    Accepts optional JSON payload: {"sample_size": int} for small-sample testing.
    """
    try:
        data = request.get_json() or {}
        sample_size = data.get("sample_size")
        if sample_size is not None:
            sample_size = int(sample_size)

        result = dataset_service.validate_rsna_dataset(
            sample_size=sample_size,
            save_splits=True
        )

        if result.get("status") == "missing_dataset":
            return jsonify({
                "status": "missing_dataset",
                "message": result["message"],
                "data": None
            }), 404
        elif result.get("status") == "error":
            return jsonify({"status": "error", "message": result["message"]}), 400

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Validation failed: {str(e)}"}), 500


@dataset_bp.route("/dataset/samples", methods=["GET"])
def get_sample_records():
    """
    Returns up to 'limit' paired records with metadata for the Dataset Explorer gallery.
    """
    try:
        limit = int(request.args.get("limit", 12))
        samples = dataset_service.get_sample_gallery(limit=limit)
        return jsonify({"status": "success", "count": len(samples), "data": samples}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@dataset_bp.route("/dataset/image/<img_id>", methods=["GET"])
def get_dataset_image(img_id):
    """
    Safely serves an X-ray image file for display in the UI.
    Includes path traversal prevention.
    """
    # Sanitize img_id to prevent directory traversal
    clean_id = os.path.basename(img_id).strip()
    raw_dir = config.DATA_RAW_DIR.resolve()

    # Search for image file
    candidates = []
    for ext in ("", ".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        test_path = (raw_dir / f"{clean_id}{ext}").resolve()
        if test_path.is_relative_to(raw_dir) and test_path.is_file():
            candidates.append(test_path)

    if not candidates:
        # Search subdirectories
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            for p in raw_dir.glob(f"**/{clean_id}{ext}"):
                res_p = p.resolve()
                if res_p.is_relative_to(raw_dir) and res_p.is_file():
                    candidates.append(res_p)
                    break
            if candidates:
                break

    if not candidates:
        abort(404, description=f"Image for ID '{clean_id}' not found.")

    return send_file(candidates[0], mimetype="image/png")


@dataset_bp.route("/dataset/splits", methods=["GET"])
def get_splits_summary():
    """
    Returns the train/validation/test split summary and leakage verification status.
    """
    splits_file = config.DATA_METADATA_DIR / "splits.json"
    if not splits_file.exists():
        return jsonify({
            "status": "not_created",
            "message": "Splits have not been generated yet. Run dataset validation first."
        }), 200

    try:
        with open(splits_file, "r") as f:
            manifest = json.load(f)

        # Do not send the huge list of IDs by default, send counts and summary
        return jsonify({
            "status": "success",
            "data": {
                "random_seed": manifest.get("random_seed"),
                "split_ratio": manifest.get("split_ratio"),
                "total_samples": manifest.get("total_samples"),
                "train_count": manifest.get("train_count"),
                "val_count": manifest.get("val_count"),
                "test_count": manifest.get("test_count"),
                "leakage_verified": manifest.get("leakage_verified"),
                "leakage_details": manifest.get("leakage_details"),
                "class_breakdown": manifest.get("class_breakdown")
            }
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
