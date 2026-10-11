"""
SQLAlchemy ORM models for SQLite database storage:
- Experiment: tracks the 12 matrix experiments and their metrics
- DatasetSummary: tracks the state of the RSNA dataset validation
- PredictionLog: tracks inference queries with inputs, outputs, and timestamps
"""

from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc)


from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, DateTime, JSON
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String(64), primary_key=True)  # e.g., "exp_P0_baseline_resnet18"
    preprocessing_id = Column(String(32), nullable=False)
    preprocessing_name = Column(String(64), nullable=False)
    model_name = Column(String(32), nullable=False)
    model_display_name = Column(String(64), nullable=False)
    status = Column(String(20), default="NOT_STARTED")  # NOT_STARTED, RUNNING, COMPLETED, FAILED
    is_smoke_test = Column(Boolean, default=False)

    # Dataset splits
    train_samples = Column(Integer, default=0)
    val_samples = Column(Integer, default=0)
    test_samples = Column(Integer, default=0)

    # Hyperparameters
    image_size = Column(Integer, default=224)
    batch_size = Column(Integer, default=16)
    learning_rate = Column(Float, default=0.0005)
    epochs = Column(Integer, default=10)
    epochs_completed = Column(Integer, default=0)
    optimizer = Column(String(32), default="Adam")

    # Timing & Complexity
    training_time_sec = Column(Float, default=0.0)
    inference_time_ms = Column(Float, default=0.0)
    total_parameters = Column(Integer, default=0)
    trainable_parameters = Column(Integer, default=0)

    # Core Evaluation Metrics
    val_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    macro_precision = Column(Float, nullable=True)
    macro_recall = Column(Float, nullable=True)
    macro_f1 = Column(Float, nullable=True)

    # Rich Metrics stored as JSON serialized strings or JSON fields
    per_class_metrics = Column(JSON, nullable=True)     # [{class_id, precision, recall, f1, support}]
    confusion_matrix = Column(JSON, nullable=True)      # 4x4 matrix
    training_history = Column(JSON, nullable=True)      # epoch-by-epoch loss & val acc

    # File artifacts
    checkpoint_path = Column(String(256), nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    def to_dict(self):
        return {
            "id": self.id,
            "preprocessing_id": self.preprocessing_id,
            "preprocessing_name": self.preprocessing_name,
            "model_name": self.model_name,
            "model_display_name": self.model_display_name,
            "status": self.status,
            "is_smoke_test": self.is_smoke_test,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "epochs_completed": self.epochs_completed,
            "optimizer": self.optimizer,
            "training_time_sec": self.training_time_sec,
            "inference_time_ms": self.inference_time_ms,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "val_accuracy": self.val_accuracy,
            "test_accuracy": self.test_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "per_class_metrics": self.per_class_metrics,
            "confusion_matrix": self.confusion_matrix,
            "training_history": self.training_history,
            "checkpoint_path": self.checkpoint_path,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DatasetSummary(Base):
    __tablename__ = "dataset_summary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    is_detected = Column(Boolean, default=False)
    csv_found = Column(Boolean, default=False)
    csv_path = Column(String(256), nullable=True)
    images_dir = Column(String(256), nullable=True)
    total_images_found = Column(Integer, default=0)
    total_csv_records = Column(Integer, default=0)
    valid_paired_records = Column(Integer, default=0)
    missing_images_count = Column(Integer, default=0)
    invalid_records_count = Column(Integer, default=0)
    duplicate_records_count = Column(Integer, default=0)
    out_of_range_count = Column(Integer, default=0)
    corrupted_images_count = Column(Integer, default=0)

    # Detailed matching stats
    unmatched_csv_records = Column(JSON, nullable=True)
    unmatched_image_files = Column(JSON, nullable=True)

    # Distribution Stats
    age_min = Column(Float, nullable=True)
    age_max = Column(Float, nullable=True)
    age_mean = Column(Float, nullable=True)
    age_std = Column(Float, nullable=True)
    age_median = Column(Float, nullable=True)
    detailed_age_stats = Column(JSON, nullable=True)  # Q1, Q3, IQR, histogram bins

    # Class & Gender distribution as JSON
    class_distribution = Column(JSON, nullable=True)
    gender_distribution = Column(JSON, nullable=True)
    class_imbalance_ratio = Column(Float, nullable=True)

    # Dataset split metadata
    split_stats = Column(JSON, nullable=True)  # train, val, test sample counts & leakage verification

    status_message = Column(Text, nullable=True)
    validated_at = Column(DateTime, default=utc_now)

    def to_dict(self):
        return {
            "id": self.id,
            "is_detected": self.is_detected,
            "csv_found": self.csv_found,
            "csv_path": self.csv_path,
            "images_dir": self.images_dir,
            "total_images_found": self.total_images_found,
            "total_csv_records": self.total_csv_records,
            "valid_paired_records": self.valid_paired_records,
            "missing_images_count": self.missing_images_count,
            "invalid_records_count": self.invalid_records_count,
            "duplicate_records_count": self.duplicate_records_count,
            "out_of_range_count": self.out_of_range_count,
            "corrupted_images_count": self.corrupted_images_count,
            "unmatched_csv_records": self.unmatched_csv_records,
            "unmatched_image_files": self.unmatched_image_files,
            "age_stats": {
                "min": self.age_min,
                "max": self.age_max,
                "mean": self.age_mean,
                "std": self.age_std,
                "median": self.age_median,
                "detailed": self.detailed_age_stats
            },
            "class_distribution": self.class_distribution,
            "class_imbalance_ratio": self.class_imbalance_ratio,
            "gender_distribution": self.gender_distribution,
            "split_stats": self.split_stats,
            "status_message": self.status_message,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None
        }


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=utc_now)
    image_name = Column(String(128), nullable=False)
    model_name = Column(String(32), nullable=False)
    preprocessing_id = Column(String(32), nullable=False)
    predicted_class = Column(Integer, nullable=False)
    predicted_label = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)  # dict of class probabilities
    inference_time_ms = Column(Float, default=0.0)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "image_name": self.image_name,
            "model_name": self.model_name,
            "preprocessing_id": self.preprocessing_id,
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "inference_time_ms": self.inference_time_ms
        }
