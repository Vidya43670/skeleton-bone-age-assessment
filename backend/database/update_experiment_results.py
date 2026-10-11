import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "backend" / "database" / "bone_age.db"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


# 12 experiments
experiments = [
    ("resnet18", "P0", 45.95, "resnet18_P0_75percent_best.pth"),
    ("resnet18", "P1", 48.01, "resnet18_P1_75percent_best.pth"),
    ("resnet18", "P2", 43.30, "resnet18_P2_75percent_best.pth"),
    ("resnet18", "P3", 50.61, "resnet18_P3_75percent_best.pth"),

    ("densenet121", "P0", 45.95, "densenet121_P0_75percent_best.pth"),
    ("densenet121", "P1", 48.01, "densenet121_P1_75percent_best.pth"),
    ("densenet121", "P2", 43.30, "densenet121_P2_75percent_best.pth"),
    ("densenet121", "P3", 50.61, "densenet121_P3_75percent_best.pth"),

    ("efficientnet_b0", "P0", 41.50, "efficientnet_b0_P0_75percent_best.pth"),
    ("efficientnet_b0", "P1", 37.90, "efficientnet_b0_P1_75percent_best.pth"),
    ("efficientnet_b0", "P2", 44.68, "efficientnet_b0_P2_75percent_best.pth"),
    ("efficientnet_b0", "P3", 41.93, "efficientnet_b0_P3_75percent_best.pth"),
]


def get_result_file(model, preprocessing):
    return RESULTS_DIR / f"{model}_{preprocessing}_75percent_results.json"


def get_experiment_id(model, preprocessing):
    mapping = {
        "P0": "exp_P0_baseline",
        "P1": "exp_P1_clahe",
        "P2": "exp_P2_denoising",
        "P3": "exp_P3_sharpening",
    }

    return f"{mapping[preprocessing]}_{model}"


def main():

    print("=" * 70)
    print("UPDATING DATABASE WITH REAL EXPERIMENT RESULTS")
    print("=" * 70)

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    updated = 0

    for model, preprocessing, val_accuracy_percent, checkpoint_name in experiments:

        result_file = get_result_file(model, preprocessing)
        checkpoint_file = CHECKPOINT_DIR / checkpoint_name
        experiment_id = get_experiment_id(model, preprocessing)

        print()
        print(f"Updating: {model} + {preprocessing}")

        if not result_file.exists():
            print(f"  ERROR: Result file missing: {result_file}")
            continue

        if not checkpoint_file.exists():
            print(f"  ERROR: Checkpoint missing: {checkpoint_file}")
            continue

        with open(result_file, "r", encoding="utf-8") as f:
            result = json.load(f)

        cursor.execute(
            """
            UPDATE experiments
            SET
                status = ?,
                is_smoke_test = ?,
                train_samples = ?,
                val_samples = ?,
                test_samples = ?,
                image_size = ?,
                batch_size = ?,
                learning_rate = ?,
                epochs = ?,
                epochs_completed = ?,
                optimizer = ?,
                val_accuracy = ?,
                test_accuracy = ?,
                macro_precision = ?,
                macro_recall = ?,
                macro_f1 = ?,
                confusion_matrix = ?,
                checkpoint_path = ?,
                error_message = NULL
            WHERE id = ?
            """,
            (
                "COMPLETED",
                0,
                6609,
                1889,
                result["test_samples"],
                224,
                4,
                0.001,
                3,
                3,
                "Adam",
                None,
                result["accuracy"],
                result["macro_precision"],
                result["macro_recall"],
                result["macro_f1"],
                json.dumps(result["confusion_matrix"]),
                str(checkpoint_file),
                experiment_id,
            ),
        )

        if cursor.rowcount == 1:
            updated += 1
            print(f"  Test Accuracy : {result['accuracy'] * 100:.2f}%")
            print(f"  Macro F1      : {result['macro_f1'] * 100:.2f}%")
            print(f"  Status        : COMPLETED")
        else:
            print(f"  WARNING: Database record not found: {experiment_id}")

    connection.commit()
    connection.close()

    print()
    print("=" * 70)
    print(f"DATABASE UPDATE COMPLETE: {updated}/12 EXPERIMENTS")
    print("=" * 70)


if __name__ == "__main__":
    main()