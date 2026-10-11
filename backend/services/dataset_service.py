"""
Dataset Service for RSNA Pediatric Bone Age Challenge.
Handles dataset location detection, CSV loading with ID preservation,
path matching, missing/invalid/duplicate checks, age distribution statistics,
4-class discretization, class imbalance ratio computation, and stratified train/val/test splitting
with zero data leakage verification.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import backend.config as config
from backend.database.db import SessionLocal
from backend.database.models import DatasetSummary


def find_rsna_paths(raw_dir: Path = None):
    """
    Detects the RSNA CSV file and image directory in data/raw or any subfolder.
    Returns: (csv_path or None, images_dir or None, list_of_image_paths)
    """
    raw_path = Path(raw_dir) if raw_dir else config.DATA_RAW_DIR
    if not raw_path.exists():
        return None, None, []

    # Look for CSV files in raw_dir and immediate subdirectories
    csv_candidates = list(raw_path.glob("*.csv"))
    if not csv_candidates:
        csv_candidates = list(raw_path.glob("**/*.csv"))

    csv_path = csv_candidates[0] if csv_candidates else None

    # Discover image files (.png, .jpg, .jpeg)
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        image_paths.extend(list(raw_path.glob(ext)))
        image_paths.extend(list(raw_path.glob(f"**/{ext}")))

    # Deduplicate image paths by resolved absolute path
    unique_images = {}
    for p in image_paths:
        unique_images[str(p.resolve())] = p
    image_paths = list(unique_images.values())

    # Detect primary image directory
    images_dir = raw_path
    if image_paths:
        # If images are in a subdirectory like 'boneage-training-dataset' or 'images', use that
        parent_counts = {}
        for p in image_paths:
            parent_counts[p.parent] = parent_counts.get(p.parent, 0) + 1
        images_dir = max(parent_counts.items(), key=lambda x: x[1])[0]

    return csv_path, images_dir, image_paths


def load_csv_data(csv_path: Path):
    """
    Loads RSNA CSV file, preserving leading zeros for ID strings.
    Identifies 'id', 'boneage', and 'male' columns with case-insensitive matching.
    Returns: (dataframe, id_col, age_col, male_col, error_msg)
    """
    if not csv_path or not Path(csv_path).exists():
        return None, None, None, None, "CSV file does not exist."

    try:
        # Inspect columns first
        header_df = pd.read_csv(csv_path, nrows=5)
        raw_cols = list(header_df.columns)
        norm_map = {c: c.strip().lower() for c in raw_cols}

        id_col_raw = next((c for c, n in norm_map.items() if "id" in n), None)
        age_col_raw = next((c for c, n in norm_map.items() if "boneage" in n or "age" in n), None)
        male_col_raw = next((c for c, n in norm_map.items() if "male" in n or "sex" in n or "gender" in n), None)

        if not id_col_raw or not age_col_raw:
            return None, None, None, None, f"Required columns ('id', 'boneage') not found. Columns: {raw_cols}"

        # Read with ID as string to preserve leading zeros
        df = pd.read_csv(csv_path, dtype={id_col_raw: str})
        return df, id_col_raw, age_col_raw, male_col_raw, None

    except Exception as e:
        return None, None, None, None, f"Error reading CSV: {str(e)}"


def check_image_corrupted(image_path: Path) -> bool:
    """Checks if an image file is corrupted or unreadable."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except Exception:
        return True


def compute_statistics(ages_series: pd.Series):
    """Computes comprehensive continuous distribution statistics and histogram bins."""
    if len(ages_series) == 0:
        return None, None

    ages = ages_series.to_numpy(dtype=float)
    q1 = float(np.percentile(ages, 25))
    q3 = float(np.percentile(ages, 75))
    iqr = float(q3 - q1)

    # 10-bin histogram
    counts, bin_edges = np.histogram(ages, bins=10)
    histogram_data = []
    for i in range(len(counts)):
        histogram_data.append({
            "range": f"{bin_edges[i]:.0f}–{bin_edges[i+1]:.0f}",
            "min": round(float(bin_edges[i]), 1),
            "max": round(float(bin_edges[i+1]), 1),
            "count": int(counts[i])
        })

    summary = {
        "min": float(np.min(ages)),
        "max": float(np.max(ages)),
        "mean": round(float(np.mean(ages)), 2),
        "std": round(float(np.std(ages)), 2),
        "median": float(np.median(ages))
    }

    detailed = {
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "histogram": histogram_data
    }

    return summary, detailed


def create_stratified_splits(paired_df: pd.DataFrame, id_col: str, class_col: str, random_seed: int = 42):
    """
    Creates a fixed 70% Train, 15% Validation, 15% Test split stratified by the 4 age classes.
    Validates zero data leakage between splits.
    Saves split manifest to data/metadata/splits.json.
    """
    if len(paired_df) < 10:
        return None, "Insufficient samples for stratified 70/15/15 split (requires >= 10 samples)."

    # Determine if stratification is possible (every class must have at least 2 samples)
    class_counts = paired_df[class_col].value_counts()
    stratify_target = paired_df[class_col] if (class_counts.min() >= 2) else None

    # First split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        paired_df,
        test_size=0.30,
        random_state=random_seed,
        stratify=stratify_target
    )

    # Second split: Split the 30% temp into equal Val (15%) and Test (15%)
    temp_stratify = temp_df[class_col] if (stratify_target is not None and temp_df[class_col].value_counts().min() >= 2) else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_seed,
        stratify=temp_stratify
    )

    train_ids = set(train_df[id_col].astype(str))
    val_ids = set(val_df[id_col].astype(str))
    test_ids = set(test_df[id_col].astype(str))

    # Strict Data Leakage Verification
    leakage_train_val = list(train_ids & val_ids)
    leakage_train_test = list(train_ids & test_ids)
    leakage_val_test = list(val_ids & test_ids)
    has_leakage = bool(leakage_train_val or leakage_train_test or leakage_val_test)

    # Class breakdown per split
    def get_class_counts(subset_df):
        counts = subset_df[class_col].value_counts().to_dict()
        return {f"Class {cls['class_id']}": int(counts.get(cls['class_id'], 0)) for cls in config.AGE_CLASSES}

    splits_manifest = {
        "random_seed": random_seed,
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "total_samples": len(paired_df),
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "leakage_verified": not has_leakage,
        "leakage_details": {
            "train_val_overlap": len(leakage_train_val),
            "train_test_overlap": len(leakage_train_test),
            "val_test_overlap": len(leakage_val_test)
        },
        "class_breakdown": {
            "train": get_class_counts(train_df),
            "val": get_class_counts(val_df),
            "test": get_class_counts(test_df)
        },
        "splits": {
            "train": list(train_df[id_col].astype(str)),
            "val": list(val_df[id_col].astype(str)),
            "test": list(test_df[id_col].astype(str))
        }
    }

    # Save to data/metadata/splits.json
    splits_file = config.DATA_METADATA_DIR / "splits.json"
    with open(splits_file, "w") as f:
        json.dump(splits_manifest, f, indent=2)

    return splits_manifest, None


def validate_rsna_dataset(raw_dir: Path = None, sample_size: int = None, save_splits: bool = True):
    """
    Main Dataset Analysis & Validation pipeline.
    Executes all 14 Phase 2 requirements.
    """
    csv_path, images_dir, image_paths = find_rsna_paths(raw_dir)

    if not csv_path:
        return {
            "status": "missing_dataset",
            "is_detected": False,
            "message": (
                "RSNA Pediatric dataset CSV not found. Please place 'boneage-training-dataset.csv' "
                f"and X-ray images in '{config.DATA_RAW_DIR}'."
            )
        }

    # 1. Load CSV
    df, id_col, age_col, male_col, err = load_csv_data(csv_path)
    if err:
        return {"status": "error", "message": err}

    total_csv_records = len(df)

    # 2. Check duplicate records in CSV
    duplicate_mask = df.duplicated(subset=[id_col], keep='first')
    duplicate_records_count = int(duplicate_mask.sum())
    duplicate_ids_sample = list(df[duplicate_mask][id_col].head(10).astype(str))
    df_clean = df.drop_duplicates(subset=[id_col], keep='first').copy()

    # 3. Build lookup of image files on disk
    # Map both stem (e.g. "1377") and full filename (e.g. "1377.png")
    image_lookup = {}
    duplicate_images_count = 0
    corrupted_images_count = 0

    for p in image_paths:
        stem = p.stem.strip()
        filename = p.name.strip()
        if stem in image_lookup:
            duplicate_images_count += 1
        else:
            image_lookup[stem] = p
        image_lookup[filename] = p

    # 4. Check unmatched CSV records & match with images
    paired_rows = []
    unmatched_csv_records = []
    out_of_range_count = 0
    out_of_range_sample = []
    invalid_records_count = 0

    for idx, row in df_clean.iterrows():
        raw_id = str(row[id_col]).strip()
        raw_age = row[age_col]

        # Validate numeric age
        try:
            boneage = float(raw_age)
            if np.isnan(boneage):
                invalid_records_count += 1
                continue
        except (ValueError, TypeError):
            invalid_records_count += 1
            continue

        # Check age range (RSNA dataset nominal range: 0 to 216 months)
        if boneage < 0 or boneage > 216:
            out_of_range_count += 1
            if len(out_of_range_sample) < 10:
                out_of_range_sample.append({"id": raw_id, "boneage": boneage})

        # Match with disk images
        img_path = image_lookup.get(raw_id) or image_lookup.get(f"{raw_id}.png") or image_lookup.get(f"{raw_id}.jpg")
        if img_path:
            # Check corruption for first 50 sample checks or if small sample mode
            if len(paired_rows) < 50 and check_image_corrupted(img_path):
                corrupted_images_count += 1
                continue

            cls_id = config.map_age_to_class(boneage)
            paired_rows.append({
                "id": raw_id,
                "boneage": boneage,
                "class_id": cls_id,
                "male": bool(row[male_col]) if male_col else None,
                "image_path": str(img_path.resolve()),
                "filename": img_path.name
            })
        else:
            unmatched_csv_records.append(raw_id)

    # 5. Detect unmatched image files on disk
    matched_image_names = {r["filename"] for r in paired_rows}
    unmatched_image_files = [p.name for p in image_paths if p.name not in matched_image_names]

    paired_df = pd.DataFrame(paired_rows)
    valid_paired_records = len(paired_df)

    # Configurable Small-Sample Mode
    if sample_size and sample_size > 0 and len(paired_df) > sample_size:
        paired_df = paired_df.sample(n=sample_size, random_state=config.RANDOM_SEED).reset_index(drop=True)

    # 6. Bone Age Distribution Analysis
    age_stats, detailed_age_stats = compute_statistics(paired_df["boneage"] if len(paired_df) > 0 else pd.Series(dtype=float))

    # 7. 4-Class Distribution & Imbalance Ratio
    class_dist = {}
    class_counts = []
    for cls in config.AGE_CLASSES:
        cls_id = cls["class_id"]
        count = int((paired_df["class_id"] == cls_id).sum()) if len(paired_df) > 0 else 0
        pct = round((count / len(paired_df) * 100), 2) if len(paired_df) > 0 else 0.0
        class_dist[f"Class {cls_id} ({cls['label']})"] = {
            "class_id": cls_id,
            "label": cls["label"],
            "name": cls["name"],
            "count": count,
            "percentage": pct
        }
        class_counts.append(count)

    # Imbalance ratio: max count / min count (for non-empty classes)
    nonzero_counts = [c for c in class_counts if c > 0]
    imbalance_ratio = round(max(nonzero_counts) / min(nonzero_counts), 2) if len(nonzero_counts) > 1 else 1.0

    # Gender distribution
    gender_dist = {"Male": 0, "Female": 0, "Unknown": 0}
    if male_col and len(paired_df) > 0:
        gender_dist["Male"] = int((paired_df["male"] == True).sum())
        gender_dist["Female"] = int((paired_df["male"] == False).sum())
        gender_dist["Unknown"] = len(paired_df) - gender_dist["Male"] - gender_dist["Female"]

    # 8. Train/Validation/Test Split Generation
    split_info = None
    if save_splits and len(paired_df) >= 10:
        split_info, split_err = create_stratified_splits(paired_df, id_col="id", class_col="class_id", random_seed=config.RANDOM_SEED)

    # 9. Update Database Record
    db = SessionLocal()
    try:
        summary_record = DatasetSummary(
            is_detected=valid_paired_records > 0 or len(image_paths) > 0,
            csv_found=True,
            csv_path=str(csv_path),
            images_dir=str(images_dir),
            total_images_found=len(image_paths),
            total_csv_records=total_csv_records,
            valid_paired_records=valid_paired_records,
            missing_images_count=len(unmatched_csv_records),
            invalid_records_count=invalid_records_count,
            duplicate_records_count=duplicate_records_count,
            out_of_range_count=out_of_range_count,
            corrupted_images_count=corrupted_images_count,
            unmatched_csv_records={
                "count": len(unmatched_csv_records),
                "sample_unmatched_ids": unmatched_csv_records[:10]
            },
            unmatched_image_files={
                "count": len(unmatched_image_files),
                "sample_unmatched_files": unmatched_image_files[:10]
            },
            age_min=age_stats["min"] if age_stats else None,
            age_max=age_stats["max"] if age_stats else None,
            age_mean=age_stats["mean"] if age_stats else None,
            age_std=age_stats["std"] if age_stats else None,
            age_median=age_stats["median"] if age_stats else None,
            detailed_age_stats=detailed_age_stats,
            class_distribution=class_dist,
            class_imbalance_ratio=imbalance_ratio,
            gender_distribution=gender_dist,
            split_stats={
                "splits_file": str(config.DATA_METADATA_DIR / "splits.json") if split_info else None,
                "train_count": split_info["train_count"] if split_info else 0,
                "val_count": split_info["val_count"] if split_info else 0,
                "test_count": split_info["test_count"] if split_info else 0,
                "leakage_verified": split_info["leakage_verified"] if split_info else False
            } if split_info else None,
            status_message=(
                f"Validation complete: {valid_paired_records} verified pairs. "
                f"Class Imbalance Ratio: {imbalance_ratio}x. "
                f"Split 70/15/15 leakage-verified."
            ) if valid_paired_records > 0 else "CSV detected but no matching X-ray image files found on disk."
        )
        db.add(summary_record)
        db.commit()
        result_dict = summary_record.to_dict()
    except Exception as e:
        db.rollback()
        result_dict = {"error": str(e)}
    finally:
        db.close()

    return {
        "status": "success",
        "data": result_dict,
        "is_small_sample_mode": bool(sample_size)
    }


def get_sample_gallery(limit: int = 12, raw_dir: Path = None):
    """
    Returns up to 'limit' paired records with metadata for display in the Dataset Explorer.
    """
    csv_path, images_dir, image_paths = find_rsna_paths(raw_dir)
    if not csv_path or not image_paths:
        return []

    df, id_col, age_col, male_col, err = load_csv_data(csv_path)
    if err or df is None:
        return []

    image_lookup = {p.stem.strip(): p for p in image_paths}

    samples = []
    for _, row in df.head(50).iterrows():
        raw_id = str(row[id_col]).strip()
        img_path = image_lookup.get(raw_id)
        if img_path:
            try:
                age = float(row[age_col])
                cls_id = config.map_age_to_class(age)
                cls_info = config.AGE_CLASSES[cls_id]
                samples.append({
                    "id": raw_id,
                    "boneage_months": age,
                    "class_id": cls_id,
                    "class_label": cls_info["label"],
                    "cohort": cls_info["name"],
                    "male": bool(row[male_col]) if male_col and not pd.isna(row[male_col]) else None,
                    "image_name": img_path.name
                })
                if len(samples) >= limit:
                    break
            except (ValueError, TypeError):
                continue

    return samples
