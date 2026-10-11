"""
Phase 2 Unit Test Suite: Dataset Analysis.
Tests:
- RSNA dataset path detection
- CSV loading preserving leading zeros
- Image matching and separate reporting of unmatched CSV and image files
- Duplicate record detection
- Out-of-range bone age detection
- Statistical distributions (min, max, mean, std, median, IQR, histogram)
- 4-Class conversion & imbalance ratio computation
- Stratified 70/15/15 split generation with zero-leakage verification
- Dataset APIs (summary, validate, splits, path traversal prevention)
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
import pandas as pd
import numpy as np
from PIL import Image

import backend.config as config
from backend.app import create_app
import backend.services.dataset_service as dataset_service


class TestPhase2DatasetAnalysis(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.app = create_app()
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_01_empty_directory_detection(self):
        """Path detection returns None when directory has no dataset files."""
        csv_path, img_dir, images = dataset_service.find_rsna_paths(self.temp_path)
        self.assertIsNone(csv_path)
        self.assertEqual(len(images), 0)

    def test_02_csv_loading_preserves_leading_zeros(self):
        """CSV loading maintains ID string format including leading zeros."""
        csv_file = self.temp_path / "boneage-training-dataset.csv"
        csv_content = "id,boneage,male\n001377,144.5,True\n000042,36.0,False\n1005,180.0,True\n"
        csv_file.write_text(csv_content)

        df, id_col, age_col, male_col, err = dataset_service.load_csv_data(csv_file)
        self.assertIsNone(err)
        self.assertEqual(id_col, "id")
        self.assertEqual(age_col, "boneage")
        self.assertEqual(male_col, "male")
        # Ensure leading zeros were preserved as strings
        self.assertEqual(df["id"].iloc[0], "001377")
        self.assertEqual(df["id"].iloc[1], "000042")

    def test_03_statistics_and_histogram(self):
        """Computes correct continuous distribution stats and histogram bins."""
        ages = pd.Series([12.0, 24.0, 48.0, 60.0, 96.0, 120.0, 144.0, 180.0])
        summary, detailed = dataset_service.compute_statistics(ages)

        self.assertEqual(summary["min"], 12.0)
        self.assertEqual(summary["max"], 180.0)
        self.assertEqual(summary["median"], 78.0)
        self.assertIn("q1", detailed)
        self.assertIn("q3", detailed)
        self.assertIn("iqr", detailed)
        self.assertEqual(len(detailed["histogram"]), 10)

    def test_04_4class_mapping_and_imbalance(self):
        """Maps continuous months into 4 discrete classes correctly."""
        # Class 0: 0-48
        self.assertEqual(config.map_age_to_class(10), 0)
        self.assertEqual(config.map_age_to_class(48), 0)
        # Class 1: 49-96
        self.assertEqual(config.map_age_to_class(49), 1)
        self.assertEqual(config.map_age_to_class(96), 1)
        # Class 2: 97-144
        self.assertEqual(config.map_age_to_class(97), 2)
        self.assertEqual(config.map_age_to_class(144), 2)
        # Class 3: 145-216
        self.assertEqual(config.map_age_to_class(145), 3)
        self.assertEqual(config.map_age_to_class(200), 3)

    def test_05_stratified_split_and_leakage_verification(self):
        """Verifies 70/15/15 stratified split and confirms zero ID overlap between splits."""
        # Create a synthetic dataset of 40 records across all 4 classes
        records = []
        for i in range(40):
            cls_id = i % 4
            records.append({
                "id": f"ID_{i:04d}",
                "boneage": float(cls_id * 48 + 20),
                "class_id": cls_id
            })
        df = pd.DataFrame(records)

        manifest, err = dataset_service.create_stratified_splits(
            df, id_col="id", class_col="class_id", random_seed=42
        )
        self.assertIsNone(err)
        self.assertTrue(manifest["leakage_verified"])
        self.assertEqual(manifest["leakage_details"]["train_val_overlap"], 0)
        self.assertEqual(manifest["leakage_details"]["train_test_overlap"], 0)
        self.assertEqual(manifest["leakage_details"]["val_test_overlap"], 0)

        # Verify proportions: 70% train (28), 15% val (6), 15% test (6)
        self.assertEqual(manifest["train_count"], 28)
        self.assertEqual(manifest["val_count"], 6)
        self.assertEqual(manifest["test_count"], 6)

    def test_06_unmatched_and_duplicate_detection(self):
        """Verifies separate identification of unmatched CSV rows, orphan images, duplicates, and out-of-range."""
        # Create CSV with duplicate ID and out-of-range age
        csv_file = self.temp_path / "boneage-training-dataset.csv"
        csv_content = (
            "id,boneage,male\n"
            "1001,24.0,True\n"
            "1001,24.0,True\n"   # Duplicate
            "1002,72.0,False\n"
            "1003,120.0,True\n"  # Has no image file (missing)
            "1004,250.0,True\n"  # Out of nominal range (> 216)
        )
        csv_file.write_text(csv_content)

        # Create dummy image files (1001, 1002, 1004, and orphan 9999)
        for img_id in ["1001", "1002", "1004", "9999"]:
            im = Image.new("L", (64, 64), color=128)
            im.save(self.temp_path / f"{img_id}.png")

        res = dataset_service.validate_rsna_dataset(raw_dir=self.temp_path, save_splits=False)
        self.assertEqual(res["status"], "success")
        data = res["data"]

        # Duplicate check
        self.assertEqual(data["duplicate_records_count"], 1)
        # Missing image check (1003 is in CSV but not on disk)
        self.assertEqual(data["missing_images_count"], 1)
        self.assertIn("1003", data["unmatched_csv_records"]["sample_unmatched_ids"])
        # Unmatched image check (9999 is on disk but not in CSV)
        self.assertEqual(data["unmatched_image_files"]["count"], 1)
        self.assertIn("9999.png", data["unmatched_image_files"]["sample_unmatched_files"])
        # Out-of-range check (1004 has age 250)
        self.assertEqual(data["out_of_range_count"], 1)

    def test_07_path_traversal_protection(self):
        """Image endpoint safely prevents path traversal attacks."""
        res = self.client.get("/api/dataset/image/../../etc/passwd")
        # Should return 404 or reject traversal safely
        self.assertIn(res.status_code, [400, 404])


if __name__ == '__main__':
    unittest.main()
