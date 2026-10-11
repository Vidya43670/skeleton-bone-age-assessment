
import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. IMPORT PROJECT MODULES
# ============================================================

from backend.training.rsna_dataset import (
    RSNABoneAgeDataset
)

from backend.models.cnn_models import (
    create_model
)


# ============================================================
# 3. TRAINING CONFIGURATION
# ============================================================

NUM_CLASSES = 4

# Smaller batch size for CPU training
BATCH_SIZE = 4

NUM_EPOCHS = 3

LEARNING_RATE = 0.001

IMAGE_SIZE = (224, 224)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 4. DATASET PATHS
# ============================================================

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "boneage-training-dataset"
)

TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.csv"
)

VALIDATION_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "validation.csv"
)


# ============================================================
# 5. OUTPUT DIRECTORY
# ============================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 6. REDUCE TRAINING DATA
# ============================================================

def create_training_subset(
    dataset,
    fraction=0.25,
    random_state=42
):
    """
    Select a percentage of the training data
    while maintaining class representation.

    Example:
        fraction=0.25 means approximately 25%.
    """

    data = dataset.data.copy()

    selected_groups = []

    for class_id in sorted(
        data["class_id"].unique()
    ):

        class_data = data[
            data["class_id"] == class_id
        ]

        sample_count = max(
            1,
            round(len(class_data) * fraction)
        )

        selected_class_data = class_data.sample(
            n=sample_count,
            random_state=random_state
        )

        selected_groups.append(
            selected_class_data
        )

    reduced_data = pd.concat(
        selected_groups,
        ignore_index=True
    )

    # Shuffle the selected records
    reduced_data = reduced_data.sample(
        frac=1.0,
        random_state=random_state
    ).reset_index(drop=True)

    dataset.data = reduced_data

    return dataset


# ============================================================
# 7. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device
):
    """
    Train the model for one complete epoch.

    Returns:
        Average training loss
        Training accuracy
    """

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_labels = []

    total_samples = 0

    total_batches = len(data_loader)

    for batch_index, (images, labels) in enumerate(
        data_loader,
        start=1
    ):

        # Move data to CPU or GPU
        images = images.to(device)

        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backward propagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Get predicted class
        _, predictions = torch.max(
            outputs,
            dim=1
        )

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        # Display progress every 100 batches
        if (
            batch_index % 100 == 0
            or batch_index == total_batches
        ):

            print(
                f"  Batch "
                f"{batch_index}/{total_batches}",
                flush=True
            )

    # Calculate average loss
    average_loss = (
        running_loss / total_samples
    )

    # Calculate accuracy
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return average_loss, accuracy


# ============================================================
# 8. VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    data_loader,
    criterion,
    device
):
    """
    Evaluate the model using the validation dataset.

    Returns:
        Average validation loss
        Validation accuracy
    """

    model.eval()

    running_loss = 0.0

    all_predictions = []

    all_labels = []

    total_samples = 0

    total_batches = len(data_loader)

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(
            data_loader,
            start=1
        ):

            # Move data to CPU or GPU
            images = images.to(device)

            labels = labels.to(device)

            # Forward propagation
            outputs = model(images)

            # Calculate validation loss
            loss = criterion(
                outputs,
                labels
            )

            # Get predicted class
            _, predictions = torch.max(
                outputs,
                dim=1
            )

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            all_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )

            all_labels.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

            # Display validation progress
            if (
                batch_index % 100 == 0
                or batch_index == total_batches
            ):

                print(
                    f"  Validation batch "
                    f"{batch_index}/{total_batches}",
                    flush=True
                )

    # Calculate average validation loss
    average_loss = (
        running_loss / total_samples
    )

    # Calculate validation accuracy
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return average_loss, accuracy


# ============================================================
# 9. MAIN TRAINING FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # 9.1 COMMAND-LINE ARGUMENTS
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Train CNN models for bone age "
            "classification"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "densenet121",
            "efficientnet_b0"
        ],
        help=(
            "CNN model to train. "
            "Default: resnet18"
        )
    )

    parser.add_argument(
        "--preprocessing",
        type=str,
        default="P0",
        choices=[
            "P0",
            "P1",
            "P2",
            "P3"
        ],
        help=(
            "Preprocessing method. "
            "P0=Baseline, "
            "P1=CLAHE, "
            "P2=Gaussian denoising, "
            "P3=Sharpening. "
            "Default: P0"
        )
    )

    args = parser.parse_args()

    MODEL_NAME = args.model

    preprocessing_id = args.preprocessing

    # 75% training data for this experiment
    TRAIN_FRACTION = 0.75

    # Unique checkpoint filename
    BEST_MODEL_PATH = (
        CHECKPOINT_DIR
        / f"{MODEL_NAME}_{preprocessing_id}_75percent_best.pth"
    )


    # --------------------------------------------------------
    # 9.2 DISPLAY TRAINING CONFIGURATION
    # --------------------------------------------------------

    print("=" * 60)

    print("BONE AGE CNN MODEL TRAINING")

    print("=" * 60)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Preprocessing method: "
        f"{preprocessing_id}"
    )

    print(
        f"Training data fraction: "
        f"{TRAIN_FRACTION * 100:.0f}%"
    )

    print(
        f"Number of classes: "
        f"{NUM_CLASSES}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Number of epochs: "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Image size: "
        f"{IMAGE_SIZE}"
    )

    print(
        f"Device: "
        f"{DEVICE}"
    )

    print(
        f"Checkpoint path: "
        f"{BEST_MODEL_PATH}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # 9.3 CHECK REQUIRED FILES AND DIRECTORIES
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{DATASET_DIR}"
        )

    if not TRAIN_CSV.exists():

        raise FileNotFoundError(
            f"Training CSV not found: "
            f"{TRAIN_CSV}"
        )

    if not VALIDATION_CSV.exists():

        raise FileNotFoundError(
            f"Validation CSV not found: "
            f"{VALIDATION_CSV}"
        )


    # --------------------------------------------------------
    # 9.4 LOAD TRAINING DATASET
    # --------------------------------------------------------

    print(
        "\nLoading training dataset...",
        flush=True
    )

    train_dataset = RSNABoneAgeDataset(
        csv_path=TRAIN_CSV,
        image_dir=DATASET_DIR,
        preprocessing_id=preprocessing_id
    )

    print(
        f"Original training samples: "
        f"{len(train_dataset)}"
    )


    # --------------------------------------------------------
    # 9.5 CREATE 25% TRAINING SUBSET
    # --------------------------------------------------------

    train_dataset = create_training_subset(
        dataset=train_dataset,
        fraction=TRAIN_FRACTION,
        random_state=42
    )

    print(
        f"Reduced training samples: "
        f"{len(train_dataset)}"
    )

    print(
        "Class distribution in reduced dataset:"
    )

    print(
        train_dataset.data["class_id"]
        .value_counts()
        .sort_index()
        .to_dict()
    )


    # --------------------------------------------------------
    # 9.6 LOAD VALIDATION DATASET
    # --------------------------------------------------------

    print(
        "\nLoading validation dataset...",
        flush=True
    )

    validation_dataset = RSNABoneAgeDataset(
        csv_path=VALIDATION_CSV,
        image_dir=DATASET_DIR,
        preprocessing_id=preprocessing_id
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )


    # --------------------------------------------------------
    # 9.7 CREATE DATA LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # --------------------------------------------------------
    # 9.8 CREATE CNN MODEL
    # --------------------------------------------------------

    print(
        "\nCreating CNN model...",
        flush=True
    )

    model = create_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    model = model.to(DEVICE)

    print(
        f"Model {MODEL_NAME} created successfully"
    )


    # --------------------------------------------------------
    # 9.9 DEFINE LOSS FUNCTION AND OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # 9.10 TRAINING LOOP
    # --------------------------------------------------------

    best_validation_accuracy = 0.0

    best_epoch = 0

    print(
        "\nStarting training...",
        flush=True
    )

    print("=" * 60)

    for epoch in range(NUM_EPOCHS):

        current_epoch = epoch + 1

        print(
            f"\nEpoch "
            f"{current_epoch}/{NUM_EPOCHS}",
            flush=True
        )


        # ----------------------------------------------------
        # Train for one epoch
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=DEVICE
            )
        )


        # ----------------------------------------------------
        # Validate the model
        # ----------------------------------------------------

        print(
            "\nStarting validation...",
            flush=True
        )

        validation_loss, validation_accuracy = (
            validate(
                model=model,
                data_loader=validation_loader,
                criterion=criterion,
                device=DEVICE
            )
        )


        # ----------------------------------------------------
        # Display epoch results
        # ----------------------------------------------------

        print(
            "\nEpoch results:"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )


        # ----------------------------------------------------
        # SAVE BEST MODEL CHECKPOINT
        # ----------------------------------------------------

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            best_epoch = current_epoch

            checkpoint = {
                "epoch": current_epoch,

                "model_name": MODEL_NAME,

                "preprocessing_id": (
                    preprocessing_id
                ),

                "training_fraction": (
                    TRAIN_FRACTION
                ),

                "num_classes": NUM_CLASSES,

                "model_state_dict": (
                    model.state_dict()
                ),

                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),

                "validation_accuracy": (
                    validation_accuracy
                )
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH
            )

            print(
                "\nBest model checkpoint saved:"
            )

            print(
                BEST_MODEL_PATH
            )


    # --------------------------------------------------------
    # 9.11 DISPLAY FINAL TRAINING SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "TRAINING COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"Model: "
        f"{MODEL_NAME}"
    )

    print(
        f"Preprocessing: "
        f"{preprocessing_id}"
    )

    print(
        f"Training data used: "
        f"{TRAIN_FRACTION * 100:.0f}%"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(
        f"Saved checkpoint: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        "=" * 60
    )


# ============================================================
# 10. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()