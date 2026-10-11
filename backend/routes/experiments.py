"""
Experiments & Evaluation Matrix API Endpoints.
GET /api/experiments
POST /api/experiments
GET /api/experiments/<id>
POST /api/experiments/<id>/train
POST /api/experiments/<id>/evaluate
GET /api/results/comparison
GET /api/results/confusion-matrix/<id>
"""

from flask import Blueprint, jsonify, request
import backend.config as config
from backend.database.db import SessionLocal
from backend.database.models import Experiment

experiments_bp = Blueprint("experiments", __name__)


@experiments_bp.route("/experiments", methods=["GET"])
def get_all_experiments():
    """Retrieve all experiments in the 12-combination research matrix."""
    db = SessionLocal()

    try:
        experiments = db.query(Experiment).all()

        return jsonify({
            "status": "success",
            "count": len(experiments),
            "data": [exp.to_dict() for exp in experiments]
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        db.close()


@experiments_bp.route("/experiments/<id>", methods=["GET"])
def get_experiment_by_id(id):
    """Retrieve details, hyperparameters, and metrics for a specific experiment."""
    db = SessionLocal()

    try:
        exp = db.query(Experiment).filter(
            Experiment.id == id
        ).first()

        if not exp:
            return jsonify({
                "status": "error",
                "message": f"Experiment '{id}' not found."
            }), 404

        return jsonify({
            "status": "success",
            "data": exp.to_dict()
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        db.close()


@experiments_bp.route("/experiments/<id>/train", methods=["POST"])
def train_experiment(id):
    """
    Trigger training for a specified experiment.
    Accepts JSON body:
    {
        "smoke_test": true|false,
        "epochs": int,
        "batch_size": int,
        "learning_rate": float
    }

    Note:
    Real training pipeline is orchestrated via backend/training engine.
    """

    db = SessionLocal()

    try:
        exp = db.query(Experiment).filter(
            Experiment.id == id
        ).first()

        if not exp:
            return jsonify({
                "status": "error",
                "message": f"Experiment '{id}' not found."
            }), 404

        data = request.get_json() or {}
        is_smoke = data.get("smoke_test", False)

        # Check if dataset exists before starting
        csv_candidates = list(config.DATA_RAW_DIR.glob("*.csv"))

        if not csv_candidates:
            return jsonify({
                "status": "error",
                "message": (
                    "Cannot begin training: No RSNA dataset found in "
                    "data/raw/. Please upload the dataset first."
                )
            }), 400

                # Do not retrain an experiment that already has stored results.
        # The existing research results should be reused.
        if exp.status == "COMPLETED" and exp.test_accuracy is not None:
            return jsonify({
                "status": "success",
                "message": (
                    f"Stored results loaded for experiment '{id}'. "
                    "No new training was started."
                ),
                "data": exp.to_dict()
            }), 200

        # Update experiment status only when a new training run is actually needed
        exp.status = "RUNNING"
        exp.is_smoke_test = is_smoke

        if "epochs" in data:
            exp.epochs = int(data["epochs"])

        if "batch_size" in data:
            exp.batch_size = int(data["batch_size"])

        if "learning_rate" in data:
            exp.learning_rate = float(data["learning_rate"])

        db.commit()

        return jsonify({
            "status": "success",
            "message": (
                f"Training initiated for experiment '{id}' "
                f"(Smoke Test: {is_smoke})."
            ),
            "data": exp.to_dict()
        }), 200

    except Exception as e:
        db.rollback()

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        db.close()


@experiments_bp.route("/results/comparison", methods=["GET"])
def get_results_comparison():
    """
    Retrieves comparative metrics table across all 12 experiments.

    Includes:
    - Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1
    - Confusion Matrix
    """

    db = SessionLocal()

    try:
        experiments = db.query(Experiment).all()
        summary = []

        for exp in experiments:

            summary.append({
                "id": exp.id,

                "preprocessing": exp.preprocessing_name,

                "model": exp.model_display_name,

                "status": exp.status,

                "is_smoke_test": exp.is_smoke_test,

                "val_accuracy": (
                    round(exp.val_accuracy * 100, 2)
                    if exp.val_accuracy is not None
                    else None
                ),

                "test_accuracy": (
                    round(exp.test_accuracy * 100, 2)
                    if exp.test_accuracy is not None
                    else None
                ),

                "macro_f1": (
                    round(exp.macro_f1, 4)
                    if exp.macro_f1 is not None
                    else None
                ),

                "macro_precision": (
                    round(exp.macro_precision, 4)
                    if exp.macro_precision is not None
                    else None
                ),

                "macro_recall": (
                    round(exp.macro_recall, 4)
                    if exp.macro_recall is not None
                    else None
                ),

                # NEW:
                # Send the stored 4x4 confusion matrix
                # directly to the frontend.
                "confusion_matrix": exp.confusion_matrix,

                "training_time_sec": (
                    round(exp.training_time_sec, 1)
                    if exp.training_time_sec
                    else None
                ),

                "epochs_completed": exp.epochs_completed
            })

        return jsonify({
            "status": "success",
            "data": summary
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        db.close()


@experiments_bp.route("/results/confusion-matrix/<id>", methods=["GET"])
def get_confusion_matrix(id):
    """Retrieve the confusion matrix and per-class metrics for an experiment."""

    db = SessionLocal()

    try:
        exp = db.query(Experiment).filter(
            Experiment.id == id
        ).first()

        if not exp:
            return jsonify({
                "status": "error",
                "message": f"Experiment '{id}' not found."
            }), 404

        if not exp.confusion_matrix:
            return jsonify({
                "status": "empty",
                "message": (
                    "No confusion matrix available yet. "
                    "Model has not been evaluated."
                )
            }), 200

        return jsonify({
            "status": "success",
            "data": {
                "experiment_id": exp.id,
                "matrix": exp.confusion_matrix,
                "per_class_metrics": exp.per_class_metrics,
                "classes": [
                    c["label"]
                    for c in config.AGE_CLASSES
                ]
            }
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        db.close()