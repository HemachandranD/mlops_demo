# housepriceai/main.py
# Main operations with Command line interface (CLI).

import json
import tempfile
import warnings
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mlflow
import optuna
import pandas as pd
import torch
import typer
from feast import FeatureStore
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from config import config
from config.config import logger
from housepriceai import data, eval, models, predict, train, utils

# Ignore warning
warnings.filterwarnings("ignore")

# Typer CLI app
app = typer.Typer()


@app.command()
def download_auxiliary_data():
    """Load auxiliary data from URL and save to local drive."""
    # Download auxiliary data
    housing_url = (
        "https://raw.githubusercontent.com/HemachandranD/mlops_demo/main/housing.json"
    )
    housing = utils.load_json_from_url(url=housing_url)

    # Save data
    housing_fp = Path(config.DATA_DIR, "housing.json")
    utils.save_dict(d=housing, filepath=housing_fp)
    logger.info("✅ Auxiliary data downloaded!")


@app.command()
def compute_features(params_fp: Path = Path(config.CONFIG_DIR, "params.json"),) -> None:
    """Compute and save features for training.
    Args:
        params_fp (Path, optional): Location of parameters (just using num_samples,
                                    num_epochs, etc.) to use for training.
                                    Defaults to `config/params.json`.
    """
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Compute features
    data.compute_features(params=params)
    logger.info("✅ Computed features!")


@app.command()
def train_model(
    params_fp: Path = Path(config.CONFIG_DIR, "params.json"),
    experiment_name: Optional[str] = "best",
    run_name: Optional[str] = "model",
) -> None:
    """Train a model using the specified parameters.
    Args:
        params_fp (Path, optional): Parameters to use for training. Defaults to `config/params.json`.
        experiment_name (str, optional): Name of the experiment to save the run to. Defaults to `best`.
        run_name (str, optional): Name of the run. Defaults to `model`.
    """
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Start run
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")

        # Train
        artifacts = train.train(params=params)

        # Log metrics
        performance = artifacts["performance"]
        logger.info(json.dumps(performance["overall"], indent=2))
        metrics = {
            # "precision": performance["overall"]["precision"],
            # "recall": performance["overall"]["recall"],
            # "f1": performance["overall"]["f1"],
            # "best_val_loss": artifacts["loss"],
            # "behavioral_score": performance["behavioral"]["score"],
            # "slices_f1": performance["slices"]["overall"]["f1"],
        }
        mlflow.log_metrics(metrics)

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["params"]), Path(dp, "params.json"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            # artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            # artifacts["tokenizer"].save(Path(dp, "tokenizer.json"))
            # torch.save(artifacts["model"].state_dict(), Path(dp, "model.pt"))
            mlflow.log_artifacts(dp)
        mlflow.log_params(vars(artifacts["params"]))


@app.command()
def predict_tags(text: str, run_id: str) -> Dict:
    """Predict tags for a give input text using a trained model.
    Warning:
        Make sure that you have a trained model first!
    Args:
        text (str): Input text to predict tags for.
        run_id (str): ID of the model run to load artifacts.
    Raises:
        ValueError: Run id doesn't exist in experiment.
    Returns:
        Predicted tags for input text.
    """
    # Predict
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))

    return prediction


@app.command()
def params(run_id: str) -> Dict:
    """Configured parametes for a specific run ID."""
    params = load_artifacts(run_id=run_id)["params"]
    logger.info(json.dumps(params, indent=2))
    return params


@app.command()
def performance(run_id: str) -> Dict:
    """Performance summary for a specific run ID."""
    performance = load_artifacts(run_id=run_id)["performance"]
    logger.info(json.dumps(performance, indent=2))
    return performance


def load_artifacts(run_id: str, device: torch.device = torch.device("cpu")) -> Dict:
    """Load artifacts for current model.
    Args:
        run_id (str): ID of the model run to load artifacts.
        device (torch.device): Device to run model on. Defaults to CPU.
    Returns:
        Artifacts needed for inference.
    """
    # Load artifacts
    artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.split("file://")[-1]
    params = Namespace(**utils.load_dict(filepath=Path(artifact_uri, "params.json")))
    performance = utils.load_dict(filepath=Path(artifact_uri, "performance.json"))

    # Initialize model
    model = models.initialize_model(params=params)
    model.load_state_dict(model_state)

    return {
        "params": params,
        "model": model,
        "performance": performance,
    }


def delete_experiment(experiment_name: str):
    """Delete an experiment with name `experiment_name`.
    Args:
        experiment_name (str): Name of the experiment.
    """
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)
    logger.info(f"✅ Deleted experiment {experiment_name}!")
