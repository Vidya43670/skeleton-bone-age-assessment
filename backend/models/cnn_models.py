
import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    densenet121,
    efficientnet_b0
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 4


# ============================================================
# RESNET18
# ============================================================

def create_resnet18(num_classes=NUM_CLASSES, pretrained=False):
    """
    Create a ResNet18 model for four-class classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained ImageNet weights.

    Returns:
        ResNet18 model.
    """

    if pretrained:
        from torchvision.models import ResNet18_Weights

        model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

    else:
        model = resnet18(
            weights=None
        )

    # Replace the original classification layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


# ============================================================
# DENSENET121
# ============================================================

def create_densenet121(num_classes=NUM_CLASSES, pretrained=False):
    """
    Create a DenseNet121 model for four-class classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained ImageNet weights.

    Returns:
        DenseNet121 model.
    """

    if pretrained:
        from torchvision.models import DenseNet121_Weights

        model = densenet121(
            weights=DenseNet121_Weights.DEFAULT
        )

    else:
        model = densenet121(
            weights=None
        )

    # Replace the original classification layer
    model.classifier = nn.Linear(
        model.classifier.in_features,
        num_classes
    )

    return model


# ============================================================
# EFFICIENTNET-B0
# ============================================================

def create_efficientnet_b0(
    num_classes=NUM_CLASSES,
    pretrained=False
):
    """
    Create an EfficientNet-B0 model for four-class classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained ImageNet weights.

    Returns:
        EfficientNet-B0 model.
    """

    if pretrained:
        from torchvision.models import EfficientNet_B0_Weights

        model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

    else:
        model = efficientnet_b0(
            weights=None
        )

    # Replace the original classification layer
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model


# ============================================================
# GENERAL MODEL FACTORY
# ============================================================

def create_model(
    model_name,
    num_classes=NUM_CLASSES,
    pretrained=False
):
    """
    Create a CNN model using its name.

    Supported models:
        resnet18
        densenet121
        efficientnet_b0

    Args:
        model_name: Name of the CNN architecture.
        num_classes: Number of classes.
        pretrained: Use pretrained weights or not.

    Returns:
        Selected CNN model.
    """

    model_name = model_name.lower().strip()

    if model_name == "resnet18":

        return create_resnet18(
            num_classes=num_classes,
            pretrained=pretrained
        )

    elif model_name == "densenet121":

        return create_densenet121(
            num_classes=num_classes,
            pretrained=pretrained
        )

    elif model_name in ["efficientnet_b0", "efficientnet-b0"]:

        return create_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained
        )

    else:

        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose resnet18, densenet121, or efficientnet_b0."
        )


# ============================================================
# TEST MODEL CREATION
# ============================================================

if __name__ == "__main__":

    print("Testing CNN model creation...")

    model_names = [
        "resnet18",
        "densenet121",
        "efficientnet_b0"
    ]

    sample_input = torch.randn(
        2, 3, 224, 224
    )

    for model_name in model_names:

        model = create_model(
            model_name=model_name,
            num_classes=4,
            pretrained=False
        )

        output = model(sample_input)

        print(
            f"{model_name}: "
            f"Input {tuple(sample_input.shape)} -> "
            f"Output {tuple(output.shape)}"
        )

    print("All CNN model tests completed.")