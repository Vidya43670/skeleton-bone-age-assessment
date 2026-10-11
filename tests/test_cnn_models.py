
import sys
from pathlib import Path

import torch


# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.models.cnn_models import get_model


def count_parameters(model):
    """Count the trainable parameters in a model."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_cnn_models():
    """Test whether all CNN models produce the expected output."""

    model_ids = [
        "resnet18",
        "densenet121",
        "efficientnet_b0"
    ]

    # Synthetic input:
    # Batch size = 2
    # Channels = 3
    # Image size = 224 x 224
    input_tensor = torch.randn(2, 3, 224, 224)

    for model_id in model_ids:

        print(f"\nTesting model: {model_id}")

        model = get_model(model_id, num_classes=4)

        model.eval()

        with torch.no_grad():
            output = model(input_tensor)

        parameter_count = count_parameters(model)

        print(f"Output shape: {output.shape}")
        print(f"Trainable parameters: {parameter_count:,}")

        # Expected output:
        # 2 images and 4 classification scores per image
        assert output.shape == (2, 4)

        assert parameter_count > 0

        print(f"{model_id} test passed.")


if __name__ == "__main__":
    test_cnn_models()

    print("\nAll CNN model tests passed.")