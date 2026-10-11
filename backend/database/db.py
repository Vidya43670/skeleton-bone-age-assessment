"""
Database engine, session management, and initialization helpers.
Uses SQLite located at backend/database/bone_age.db.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import sqlite3
import backend.config as config
from backend.database.models import Base, Experiment

# Setup SQLite engine
engine = create_engine(
    f"sqlite:///{config.DATABASE_PATH}",
    connect_args={"check_same_thread": False},
    echo=False
)

SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)


def _migrate_db():
    """Dynamically add any missing columns to existing SQLite tables without data loss."""
    if not config.DATABASE_PATH.exists():
        return

    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(dataset_summary)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        new_cols = {
            "out_of_range_count": "INTEGER DEFAULT 0",
            "corrupted_images_count": "INTEGER DEFAULT 0",
            "unmatched_csv_records": "JSON",
            "unmatched_image_files": "JSON",
            "detailed_age_stats": "JSON",
            "class_imbalance_ratio": "FLOAT",
            "split_stats": "JSON"
        }
        for col_name, col_type in new_cols.items():
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE dataset_summary ADD COLUMN {col_name} {col_type}")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB MIGRATION NOTICE]: {e}")


def init_db():
    """Create all tables, migrate columns, and seed the initial 12-combination experiment matrix."""
    Base.metadata.create_all(bind=engine)
    _migrate_db()
    seed_experiment_matrix()


def get_db():
    """Contextual dependency for DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_experiment_matrix():
    """
    Ensure all 12 experimental combinations (4 Preprocessing x 3 CNNs) exist in the database.
    Does not overwrite existing experiments.
    """
    db = SessionLocal()
    try:
        for prep_id, prep_info in config.PREPROCESSING_METHODS.items():
            for model_id, model_info in config.CNN_ARCHITECTURES.items():
                exp_id = f"exp_{prep_id}_{model_id}"
                existing = db.query(Experiment).filter(Experiment.id == exp_id).first()
                if not existing:
                    exp = Experiment(
                        id=exp_id,
                        preprocessing_id=prep_id,
                        preprocessing_name=prep_info["name"],
                        model_name=model_id,
                        model_display_name=model_info["name"],
                        status="NOT_STARTED",
                        is_smoke_test=False,
                        image_size=config.DEFAULT_TRAINING_CONFIG["image_size"],
                        batch_size=config.DEFAULT_TRAINING_CONFIG["batch_size"],
                        learning_rate=config.DEFAULT_TRAINING_CONFIG["learning_rate"],
                        epochs=config.DEFAULT_TRAINING_CONFIG["epochs"],
                        optimizer=config.DEFAULT_TRAINING_CONFIG["optimizer"]
                    )
                    db.add(exp)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB INIT ERROR]: {e}")
    finally:
        db.close()
