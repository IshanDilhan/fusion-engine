"""Shared MLflow tracking setup.

Central place for tracking URI / experiment initialization so every
training/eval script (fusion/rule_based.py, fusion/gbt.py, and future
retraining scripts) points at the same local SQLite backend store and
artifact directory instead of each reinventing this.
"""
import os

import mlflow

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DB_PATH = os.path.join(REPO_ROOT, "mlflow.db")
ARTIFACT_ROOT = os.path.join(REPO_ROOT, "mlartifacts")
TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
EXPERIMENT_NAME = "fusion-engine-intent-classification"


def init_tracking(experiment_name: str = EXPERIMENT_NAME) -> str:
    """Points MLflow at the project's local SQLite store, ensuring the
    target experiment exists, and makes it the active experiment.

    Returns the experiment_id.
    """
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = f"file://{ARTIFACT_ROOT}"
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id
