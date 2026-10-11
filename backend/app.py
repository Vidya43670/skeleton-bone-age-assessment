"""
Flask Application Entry Point for Skeleton Bone Age Assessment Research Platform.
Configures CORS, registers modular blueprints, and initializes the database.
"""

from flask import Flask, jsonify
from flask_cors import CORS
import backend.config as config
from backend.database.db import init_db

# Import route blueprints
from backend.routes.health import health_bp
from backend.routes.dataset import dataset_bp
from backend.routes.preprocessing import preprocessing_bp
from backend.routes.models_api import models_bp
from backend.routes.experiments import experiments_bp
from backend.routes.predict import predict_bp


def create_app():
    """Application factory pattern."""
    app = Flask(__name__)

    # Enable Cross-Origin Resource Sharing
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints under /api prefix
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(dataset_bp, url_prefix="/api")
    app.register_blueprint(preprocessing_bp, url_prefix="/api")
    app.register_blueprint(models_bp, url_prefix="/api")
    app.register_blueprint(experiments_bp, url_prefix="/api")
    app.register_blueprint(predict_bp, url_prefix="/api")

    # Global root route
    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "project": "Skeleton Bone Age Assessment Using Advanced Image Preprocessing and CNN-Based Classification",
            "version": "1.0.0",
            "status": "online",
            "health_endpoint": "/api/health",
            "disclaimer": config.DISCLAIMER_TEXT
        })

    # Initialize SQLite database and 12-experiment matrix
    with app.app_context():
        init_db()

    return app


app = create_app()

if __name__ == "__main__":
    print(f"[*] Starting Skeleton Bone Age Assessment API Server...")
    print(f"[*] Hardware Execution Device: {config.DEVICE_NAME} ({config.DEVICE})")
    print(f"[*] Database Path: {config.DATABASE_PATH}")
    print(f"[*] Base Workspace: {config.BASE_DIR}")
    app.run(host="0.0.0.0", port=5000, debug=True)
