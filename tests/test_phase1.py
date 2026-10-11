"""
Phase 1 Automated Test Suite:
Tests system health check, database matrix initialization,
preprocessing methods, model specifications, and 4-class mapping.
"""

import unittest
import json
import backend.config as config
from backend.app import create_app


class TestPhase1Setup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_01_health_check(self):
        """Verify GET /api/health returns 200 and expected schema."""
        res = self.client.get('/api/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("hardware", data)
        self.assertIn("database", data)
        self.assertIn("dataset", data)
        self.assertIn("research_matrix", data)
        self.assertEqual(data["database"]["status"], "connected")
        self.assertEqual(data["research_matrix"]["total_combinations"], 12)
        self.assertEqual(data["research_matrix"]["preprocessing_methods_count"], 4)
        self.assertEqual(data["research_matrix"]["cnn_architectures_count"], 3)
        self.assertEqual(data["research_matrix"]["age_classes_count"], 4)

    def test_02_preprocessing_methods(self):
        """Verify GET /api/preprocessing/methods returns all 4 methods (P0 to P3)."""
        res = self.client.get('/api/preprocessing/methods')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        methods = {m["code"] for m in data["data"]}
        self.assertEqual(methods, {"P0", "P1", "P2", "P3"})

    def test_03_models_list(self):
        """Verify GET /api/models returns 3 architectures with 4 output classes."""
        res = self.client.get('/api/models')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        models = {m["id"] for m in data["data"]}
        self.assertEqual(models, {"resnet18", "densenet121", "efficientnet_b0"})
        for m in data["data"]:
            self.assertEqual(m["adapted_num_classes"], 4)

    def test_04_experiment_matrix(self):
        """Verify GET /api/experiments returns exactly 12 combinations."""
        res = self.client.get('/api/experiments')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["count"], 12)
        # Verify all start as NOT_STARTED without fake results
        for exp in data["data"]:
            self.assertEqual(exp["status"], "NOT_STARTED")
            self.assertIsNone(exp["test_accuracy"])
            self.assertIsNone(exp["macro_f1"])

    def test_05_age_to_class_mapping(self):
        """Verify the 4-class discrete mapping function."""
        # Class 0: 0-48 months
        self.assertEqual(config.map_age_to_class(0), 0)
        self.assertEqual(config.map_age_to_class(24), 0)
        self.assertEqual(config.map_age_to_class(48), 0)

        # Class 1: 49-96 months
        self.assertEqual(config.map_age_to_class(49), 1)
        self.assertEqual(config.map_age_to_class(72), 1)
        self.assertEqual(config.map_age_to_class(96), 1)

        # Class 2: 97-144 months
        self.assertEqual(config.map_age_to_class(97), 2)
        self.assertEqual(config.map_age_to_class(120), 2)
        self.assertEqual(config.map_age_to_class(144), 2)

        # Class 3: 145-216 months
        self.assertEqual(config.map_age_to_class(145), 3)
        self.assertEqual(config.map_age_to_class(180), 3)
        self.assertEqual(config.map_age_to_class(216), 3)


if __name__ == '__main__':
    unittest.main()
