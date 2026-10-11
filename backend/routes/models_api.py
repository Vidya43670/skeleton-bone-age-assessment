"""
CNN Models API Endpoints.
GET /api/models
"""

from flask import Blueprint, jsonify
import backend.config as config

models_bp = Blueprint("models_api", __name__)


@models_bp.route("/models", methods=["GET"])
def get_models():
    """Returns available CNN architectures, theoretical parameters, and configuration specs."""
    models_info = [
        {
            "id": "resnet18",
            "name": "ResNet-18",
            "family": "Residual Networks (He et al., 2015)",
            "approx_params": "11.2M",
            "adapted_num_classes": config.NUM_CLASSES,
            "input_resolution": "224x224x3",
            "description": "Standard 18-layer residual architecture using identity shortcut mappings to prevent gradient vanishing.",
            "transfer_learning_support": True,
            "weights": "ImageNet (Torchvision DEFAULT)"
        },
        {
            "id": "densenet121",
            "name": "DenseNet-121",
            "family": "Densely Connected Networks (Huang et al., 2017)",
            "approx_params": "7.0M",
            "adapted_num_classes": config.NUM_CLASSES,
            "input_resolution": "224x224x3",
            "description": "121-layer dense architecture where each layer connects to every other layer in a feed-forward fashion for maximum feature reuse.",
            "transfer_learning_support": True,
            "weights": "ImageNet (Torchvision DEFAULT)"
        },
        {
            "id": "efficientnet_b0",
            "name": "EfficientNet-B0",
            "family": "EfficientNet (Tan & Le, 2019)",
            "approx_params": "4.0M",
            "adapted_num_classes": config.NUM_CLASSES,
            "input_resolution": "224x224x3",
            "description": "Compound scaling model that balances network depth, width, and resolution using Mobile Inverted Bottleneck Conv (MBConv) blocks.",
            "transfer_learning_support": True,
            "weights": "ImageNet (Torchvision DEFAULT)"
        }
    ]

    return jsonify({
        "status": "success",
        "data": models_info,
        "classes": config.AGE_CLASSES
    }), 200
