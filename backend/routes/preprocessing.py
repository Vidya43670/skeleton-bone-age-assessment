"""
Image Preprocessing API Endpoints.
GET /api/preprocessing/methods
POST /api/preprocessing/preview
"""

from flask import Blueprint, jsonify, request
import cv2
import numpy as np
import base64
import io
from PIL import Image
import backend.config as config

preprocessing_bp = Blueprint("preprocessing", __name__)


def apply_preprocessing(image_bgr: np.ndarray, method_id: str, custom_params: dict = None):
    """
    Applies the chosen preprocessing technique to the input BGR image.
    Supports P0 (Baseline), P1 (CLAHE), P2 (Denoising), P3 (Sharpening).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    params = custom_params or {}

    if method_id == "P0_baseline":
        # Baseline: resize to 224x224 and standard min-max / contrast normalization
        size = params.get("target_size", [224, 224])
        resized = cv2.resize(gray, (size[0], size[1]), interpolation=cv2.INTER_AREA)
        norm = cv2.normalize(resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    elif method_id == "P1_clahe":
        # CLAHE: Contrast Limited Adaptive Histogram Equalization
        clip_limit = float(params.get("clip_limit", 2.0))
        grid_size = params.get("tile_grid_size", [8, 8])
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(int(grid_size[0]), int(grid_size[1])))
        equalized = clahe.apply(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    elif method_id == "P2_denoising":
        # Denoising: Gaussian or Median filtering
        filter_type = params.get("filter_type", "gaussian")
        kernel_size = int(params.get("kernel_size", 3))
        if kernel_size % 2 == 0:
            kernel_size += 1

        if filter_type == "median":
            denoised = cv2.medianBlur(gray, kernel_size)
        else:
            sigma = float(params.get("sigma", 0.8))
            denoised = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    elif method_id == "P3_sharpening":
        # Sharpening: Unsharp Masking
        amount = float(params.get("amount", 1.5))
        radius = float(params.get("radius", 1.0))
        ksize = int(2 * round(radius * 2) + 1)
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), radius)
        # unsharp mask = original + amount * (original - blurred)
        sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    else:
        # Default fallback
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@preprocessing_bp.route("/preprocessing/methods", methods=["GET"])
def get_preprocessing_methods():
    """Returns available preprocessing methods and default parameter configurations."""
    return jsonify({
        "status": "success",
        "data": list(config.PREPROCESSING_METHODS.values())
    }), 200


@preprocessing_bp.route("/preprocessing/preview", methods=["POST"])
def preview_preprocessing():
    """
    Takes an uploaded image or sample index and returns base64 of original and processed images.
    """
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file provided in request."}), 400

    file = request.files["image"]
    method_id = request.form.get("method_id", "P0_baseline")

    if method_id not in config.PREPROCESSING_METHODS:
        return jsonify({
            "status": "error",
            "message": f"Unknown preprocessing method '{method_id}'. Available: {list(config.PREPROCESSING_METHODS.keys())}"
        }), 400

    try:
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"status": "error", "message": "Invalid or corrupted image format."}), 400

        # Apply preprocessing
        processed_bgr = apply_preprocessing(img_bgr, method_id)

        # Convert both to base64 JPEG
        _, orig_buf = cv2.imencode(".jpg", img_bgr)
        orig_b64 = base64.b64encode(orig_buf).decode("utf-8")

        _, proc_buf = cv2.imencode(".jpg", processed_bgr)
        proc_b64 = base64.b64encode(proc_buf).decode("utf-8")

        return jsonify({
            "status": "success",
            "method_id": method_id,
            "method_info": config.PREPROCESSING_METHODS[method_id],
            "original_image": f"data:image/jpeg;base64,{orig_b64}",
            "processed_image": f"data:image/jpeg;base64,{proc_b64}"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
