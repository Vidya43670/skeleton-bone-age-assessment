
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.preprocessing.image_preprocessing import apply_preprocessing
from backend.models.cnn_models import get_model


# Configuration
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)

PREPROCESSING_IDS = [
    "P0",
    "P1",
    "P2",
    "P3"
]

MODEL_IDS = [
    "resnet18",
    "densenet121",
    "efficientnet_b0"
]


def prepare_image(image, preprocessing_id):
    """
    Apply the selected preprocessing method and convert
    the grayscale image into a 3-channel PyTorch tensor.

    Input:
        image: Grayscale NumPy image
        preprocessing_id: P0, P1, P2, or P3

    Output:
        Tensor with shape (3, 224, 224)
    """

    processed_image = apply_preprocessing(
        image,
        preprocessing_id,
        image_size=IMAGE_SIZE
    )

    # Convert NumPy array into a PyTorch tensor
    image_tensor = torch.from_numpy(
        processed_image
    ).float()

    # Add channel dimension
    # (224, 224) -> (1, 224, 224)
    image_tensor = image_tensor.unsqueeze(0)

    # Convert grayscale to 3 channels
    # (1, 224, 224) -> (3, 224, 224)
    image_tensor = image_tensor.repeat(3, 1, 1)

    return image_tensor


def test_training_pipeline():
    """
    Test one training step for all 12 combinations:

    4 preprocessing methods × 3 CNN models = 12 tests
    """

    # Create a synthetic grayscale X-ray-like image
    # This is only for pipeline testing.
    synthetic_image = np.random.randint(
        0,
        256,
        size=(300, 300),
        dtype=np.uint8
    )

    # Synthetic target label
    # Class values: 0, 1, 2, or 3
    target = torch.tensor(
        [1],
        dtype=torch.long
    )

    # Loss function for multi-class classification
    loss_function = nn.CrossEntropyLoss()

    total_tests = 0

    passed_tests = 0

    print("=" * 60)
    print("CNN TRAINING PIPELINE SMOKE TEST")
    print("=" * 60)

    print(f"Preprocessing methods: {len(PREPROCESSING_IDS)}")
    print(f"CNN models: {len(MODEL_IDS)}")
    print(
        f"Total combinations: "
        f"{len(PREPROCESSING_IDS) * len(MODEL_IDS)}"
    )

    # Loop through all preprocessing methods
    for preprocessing_id in PREPROCESSING_IDS:

        print("\n" + "-" * 60)
        print(
            f"PREPROCESSING METHOD: {preprocessing_id}"
        )
        print("-" * 60)

        # Apply the selected preprocessing method
        image_tensor = prepare_image(
            synthetic_image,
            preprocessing_id
        )

        # Add batch dimension
        # (3, 224, 224) -> (1, 3, 224, 224)
        input_tensor = image_tensor.unsqueeze(0)

        print(
            f"Preprocessed tensor shape: "
            f"{input_tensor.shape}"
        )

        # Loop through all CNN models
        for model_id in MODEL_IDS:

            total_tests += 1

            print("\n" + "=" * 50)
            print(
                f"Testing combination: "
                f"{preprocessing_id} + {model_id}"
            )
            print("=" * 50)

            # Create CNN model
            model = get_model(
                model_id,
                num_classes=NUM_CLASSES
            )

            # Set model to training mode
            model.train()

            # Adam optimizer
            optimizer = optim.Adam(
                model.parameters(),
                lr=0.001
            )

            # Forward pass
            output = model(input_tensor)

            # Calculate loss
            loss = loss_function(
                output,
                target
            )

            # Clear previous gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Display results
            print(
                f"Input shape: {input_tensor.shape}"
            )

            print(
                f"Output shape: {output.shape}"
            )

            print(
                f"Loss: {loss.item():.4f}"
            )

            # Verify expected output
            assert input_tensor.shape == (
                1,
                3,
                224,
                224
            )

            assert output.shape == (
                1,
                NUM_CLASSES
            )

            assert torch.isfinite(loss).item()

            assert loss.item() >= 0

            passed_tests += 1

            print(
                f"PASS: "
                f"{preprocessing_id} + {model_id}"
            )

            # Delete model to release memory
            del model
            del optimizer

    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)

    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(
        f"Failed tests: "
        f"{total_tests - passed_tests}"
    )

    assert total_tests == 12
    assert passed_tests == 12

    print("\nAll 12 training pipeline tests passed!")


if __name__ == "__main__":
    test_training_pipeline()