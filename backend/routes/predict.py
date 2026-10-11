"""
Single Image Inference Demonstration API Endpoint.

POST /api/predict

Uses an existing trained checkpoint from the 12-experiment research matrix.
No training is performed during prediction.
"""

from flask import Blueprint, jsonify, request

import cv2
import numpy as np
import os
import torch

import backend.config as config
from backend.database.db import SessionLocal
from backend.database.models import Experiment
from backend.models.cnn_models import create_model
from backend.preprocessing.image_preprocessing import apply_preprocessing


predict_bp = Blueprint("predict", __name__)


# Research-defined age classes used by this project.
CLASS_RANGES = {
    0: {"min_months": 0, "max_months": 48, "label": "0–48 months"},
    1: {"min_months": 49, "max_months": 96, "label": "49–96 months"},
    2: {"min_months": 97, "max_months": 144, "label": "97–144 months"},
    3: {"min_months": 145, "max_months": 216, "label": "145–216 months"},
}


def preprocess_for_model(image, preprocessing_id):
    """
    Apply the same preprocessing used during training.

    Returns:
        torch.Tensor with shape (1, 3, 224, 224)
    """

    # Convert database/frontend preprocessing ID to the
    # P0/P1/P2/P3 IDs expected by apply_preprocessing().
    prep_map = {
        "P0_baseline": "P0",
        "P1_clahe": "P1",
        "P2_denoising": "P2",
        "P3_sharpening": "P3",
    }

    short_prep_id = prep_map.get(preprocessing_id)

    if short_prep_id is None:
        raise ValueError(
            f"Unsupported preprocessing ID: {preprocessing_id}"
        )

    processed = apply_preprocessing(
        image,
        short_prep_id,
        image_size=(224, 224)
    )

    # processed shape: (224, 224), values: 0-1
    tensor = torch.from_numpy(
        processed.astype(np.float32)
    )

    # Add channel dimension:
    # (224,224) -> (1,224,224)
    tensor = tensor.unsqueeze(0)

    # Training pipeline repeats grayscale channel to RGB:
    # (1,224,224) -> (3,224,224)
    tensor = tensor.repeat(3, 1, 1)

    # Add batch dimension:
    # (3,224,224) -> (1,3,224,224)
    tensor = tensor.unsqueeze(0)

    return tensor


@predict_bp.route("/predict", methods=["POST"])
def predict_bone_age():
    """
    Run real inference using an existing trained checkpoint.
    """

    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image file provided."
        }), 400

    file = request.files["image"]

    model_name = request.form.get(
        "model_name",
        "resnet18"
    )

    preprocessing_id = request.form.get(
        "preprocessing_id",
        "P0_baseline"
    )

    # Validate model
    if model_name not in config.CNN_ARCHITECTURES:
        return jsonify({
            "status": "error",
            "message": f"Invalid model '{model_name}'."
        }), 400

    # Validate preprocessing
    if preprocessing_id not in config.PREPROCESSING_METHODS:
        return jsonify({
            "status": "error",
            "message": (
                f"Invalid preprocessing '{preprocessing_id}'."
            )
        }), 400

    # Build experiment ID exactly as stored in the database.
    exp_id = f"exp_{preprocessing_id}_{model_name}"

    db = SessionLocal()

    try:
        # ---------------------------------------------------------
        # 1. Get experiment and checkpoint
        # ---------------------------------------------------------
        exp = (
            db.query(Experiment)
            .filter(Experiment.id == exp_id)
            .first()
        )

        if (
            not exp
            or exp.status != "COMPLETED"
            or not exp.checkpoint_path
            or not os.path.exists(exp.checkpoint_path)
        ):
            return jsonify({
                "status": "not_ready",
                "message": (
                    f"Model combination "
                    f"'{config.CNN_ARCHITECTURES[model_name]['name']} + "
                    f"{config.PREPROCESSING_METHODS[preprocessing_id]['name']}' "
                    f"does not have a usable completed checkpoint."
                ),
                "experiment_status": (
                    exp.status if exp else "NOT_STARTED"
                ),
                "disclaimer": config.DISCLAIMER_TEXT
            }), 400

        # ---------------------------------------------------------
        # 2. Read uploaded image
        # ---------------------------------------------------------
        image_bytes = file.read()

        if not image_bytes:
            return jsonify({
                "status": "error",
                "message": "Uploaded image is empty."
            }), 400

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            return jsonify({
                "status": "error",
                "message": (
                    "Could not decode the uploaded image. "
                    "Please use PNG or JPG."
                )
            }), 400

        # ---------------------------------------------------------
        # 3. Apply the same preprocessing used during training
        # ---------------------------------------------------------
        input_tensor = preprocess_for_model(
            image,
            preprocessing_id
        )

        # ---------------------------------------------------------
        # 4. Create the same CNN architecture
        # ---------------------------------------------------------
        model = create_model(
            model_name=model_name,
            num_classes=4,
            pretrained=False
        )

        # ---------------------------------------------------------
        # 5. Load existing trained checkpoint
        # ---------------------------------------------------------
        checkpoint = torch.load(
            exp.checkpoint_path,
            map_location="cpu"
        )

        if "model_state_dict" not in checkpoint:
            return jsonify({
                "status": "error",
                "message": (
                    "Checkpoint does not contain "
                    "model_state_dict."
                )
            }), 500

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        # ---------------------------------------------------------
        # 6. Real CNN inference
        # ---------------------------------------------------------
        with torch.no_grad():
            logits = model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

            predicted_class = int(
                torch.argmax(probabilities).item()
            )

        # ---------------------------------------------------------
        # 7. Prepare class probability output
        # ---------------------------------------------------------
        class_probabilities = []

        for class_id in range(4):
            class_info = CLASS_RANGES[class_id]

            class_probabilities.append({
                "class_id": class_id,
                "label": class_info["label"],
                "probability": round(
                    float(probabilities[class_id].item()),
                    6
                ),
                "percentage": round(
                    float(probabilities[class_id].item()) * 100,
                    2
                )
            })

        predicted_range = CLASS_RANGES[
            predicted_class
        ]

        # ---------------------------------------------------------
        # 8. Return actual prediction
        # ---------------------------------------------------------
        return jsonify({
            "status": "success",
            "message": "Inference completed successfully.",

            "prediction": {
                "predicted_class": predicted_class,
                "age_range": predicted_range["label"],
                "confidence": round(
                    float(
                        probabilities[predicted_class].item()
                    ) * 100,
                    2
                )
            },

            "class_probabilities": class_probabilities,

            "model": {
                "name": model_name,
                "display_name": (
                    config.CNN_ARCHITECTURES[
                        model_name
                    ]["name"]
                )
            },

            "preprocessing": {
                "id": preprocessing_id,
                "name": (
                    config.PREPROCESSING_METHODS[
                        preprocessing_id
                    ]["name"]
                )
            },

            "checkpoint": {
                "path": exp.checkpoint_path,
                "validation_accuracy": (
                    exp.val_accuracy
                ),
                "test_accuracy": (
                    exp.test_accuracy
                )
            },

            "disclaimer": config.DISCLAIMER_TEXT
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        db.close()