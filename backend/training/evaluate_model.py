from pathlib import Path
import sys
import json

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.training.rsna_dataset import RSNABoneAgeDataset
from backend.models.cnn_models import create_model


# ============================================================
# 2. CONFIGURATION
# ============================================================

NUM_CLASSES = 4
BATCH_SIZE = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "boneage-training-dataset"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. EXPERIMENTS 5–12
# ============================================================

EXPERIMENTS = [
    ("densenet121", "P0"),
    ("densenet121", "P1"),
    ("densenet121", "P2"),
    ("densenet121", "P3"),

    ("efficientnet_b0", "P0"),
    ("efficientnet_b0", "P1"),
    ("efficientnet_b0", "P2"),
    ("efficientnet_b0", "P3")
]


# ============================================================
# 4. LOAD TEST DATASET
# ============================================================

print("=" * 70)
print("BONE AGE CNN TEST-SET EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Test CSV: {TEST_CSV}")
print()

print("Loading test dataset...")

# Dataset will be recreated for each preprocessing method.
# This is necessary because P0/P1/P2/P3 process images differently.


# ============================================================
# 5. EVALUATE ONE EXPERIMENT
# ============================================================

def evaluate_experiment(
    experiment_number,
    model_name,
    preprocessing_method
):

    print("\n" + "=" * 70)

    print(
        f"EXPERIMENT {experiment_number}"
    )

    print(
        f"Model: {model_name}"
    )

    print(
        f"Preprocessing: {preprocessing_method}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Checkpoint path
    # --------------------------------------------------------

    checkpoint_name = (
        f"{model_name}_"
        f"{preprocessing_method}_"
        f"75percent_best.pth"
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / checkpoint_name
    )


    # --------------------------------------------------------
    # Results path
    # --------------------------------------------------------

    results_name = (
        f"{model_name}_"
        f"{preprocessing_method}_"
        f"75percent_results.json"
    )

    results_path = (
        RESULTS_DIR
        / results_name
    )


    print(
        f"Checkpoint: {checkpoint_path}"
    )


    # --------------------------------------------------------
    # Check checkpoint exists
    # --------------------------------------------------------

    if not checkpoint_path.exists():

        print(
            "\nERROR: Checkpoint not found!"
        )

        print(
            checkpoint_path
        )

        print(
            "Skipping this experiment."
        )

        return None


    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print(
        "\nLoading test dataset..."
    )

    test_dataset = RSNABoneAgeDataset(
        csv_path=TEST_CSV,
        image_dir=IMAGE_DIR,
        preprocessing_id=preprocessing_method
    )


    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )


    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # --------------------------------------------------------
    # Create CNN model
    # --------------------------------------------------------

    print(
        "\nCreating model..."
    )

    model = create_model(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        pretrained=False
    )


    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print(
        "Loading trained checkpoint..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )


    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )


    model = model.to(DEVICE)

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )


    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    all_labels = []

    all_predictions = []


    print(
        "\nEvaluating model on test data..."
    )


    with torch.no_grad():

        for batch_number, (
            images,
            labels
        ) in enumerate(test_loader, start=1):

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )


            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )


            if batch_number % 100 == 0:

                print(
                    f"  Evaluated batch "
                    f"{batch_number}/"
                    f"{len(test_loader)}"
                )


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    precision_macro = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    recall_macro = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    f1_macro = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    precision_weighted = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )


    recall_weighted = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )


    f1_weighted = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )


    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3]
    )


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3],
        zero_division=0
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "\n========== TEST RESULTS =========="
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision: "
        f"{precision_macro * 100:.2f}%"
    )

    print(
        f"Macro Recall: "
        f"{recall_macro * 100:.2f}%"
    )

    print(
        f"Macro F1 Score: "
        f"{f1_macro * 100:.2f}%"
    )

    print(
        f"Weighted Precision: "
        f"{precision_weighted * 100:.2f}%"
    )

    print(
        f"Weighted Recall: "
        f"{recall_weighted * 100:.2f}%"
    )

    print(
        f"Weighted F1 Score: "
        f"{f1_weighted * 100:.2f}%"
    )


    print(
        "\nConfusion Matrix:"
    )

    print(matrix)


    print(
        "\nClassification Report:"
    )

    print(report)


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {

        "experiment": experiment_number,

        "model": model_name,

        "preprocessing": preprocessing_method,

        "training_data_fraction": 0.75,

        "test_samples": len(test_dataset),

        "device": str(DEVICE),

        "accuracy": float(accuracy),

        "macro_precision": float(
            precision_macro
        ),

        "macro_recall": float(
            recall_macro
        ),

        "macro_f1": float(
            f1_macro
        ),

        "weighted_precision": float(
            precision_weighted
        ),

        "weighted_recall": float(
            recall_weighted
        ),

        "weighted_f1": float(
            f1_weighted
        ),

        "confusion_matrix": matrix.tolist()
    }


    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )


    print(
        "\nResults saved to:"
    )

    print(results_path)


    print(
        "\nExperiment evaluation completed."
    )


    return results


# ============================================================
# 6. RUN EXPERIMENTS 5–12
# ============================================================

all_results = []


for index, (
    model_name,
    preprocessing_method
) in enumerate(
    EXPERIMENTS,
    start=5
):

    result = evaluate_experiment(
        experiment_number=index,
        model_name=model_name,
        preprocessing_method=preprocessing_method
    )


    if result is not None:

        all_results.append(
            result
        )


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("ALL REQUESTED EVALUATIONS COMPLETED")
print("=" * 70)

print(
    f"Experiments successfully evaluated: "
    f"{len(all_results)}/8"
)

print()


for result in all_results:

    print(
        f"Experiment "
        f"{result['experiment']}: "
        f"{result['model']} + "
        f"{result['preprocessing']} | "
        f"Accuracy = "
        f"{result['accuracy'] * 100:.2f}% | "
        f"Macro F1 = "
        f"{result['macro_f1'] * 100:.2f}%"
    )


print()
print(
    "Evaluation process finished."
)
print("=" * 70)