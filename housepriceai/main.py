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
def optimize(
    params_fp: Path = Path(config.CONFIG_DIR, "params.json"),
    study_name: Optional[str] = "optimization",
    num_trials: int = 100,
) -> None:
    """Optimize a subset of hyperparameters towards an objective.
    This saves the best trial's parameters into `config/params.json`.
    Args:
        params_fp (Path, optional): Location of parameters (just using num_samples,
                                    num_epochs, etc.) to use for training.
                                    Defaults to `config/params.json`.
        study_name (str, optional): Name of the study to save trial runs under. Defaults to `optimization`.
        num_trials (int, optional): Number of trials to run. Defaults to 100.
    """
    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name=study_name, direction="maximize", pruner=pruner
    )
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="f1"
    )
    study.optimize(
        lambda trial: train.objective(params, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # All trials
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["value"], ascending=False)

    # Best trial
    logger.info(f"Best value (f1): {study.best_trial.value}")
    params = {**params.__dict__, **study.best_trial.params}
    params["threshold"] = study.best_trial.user_attrs["threshold"]
    utils.save_dict(params, params_fp, cls=NumpyEncoder)
    logger.info(json.dumps(params, indent=2, cls=NumpyEncoder))


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

        # Set tags
        tags = {}
        mlflow.set_tags(tags)

        # Log metrics
        performance = artifacts["performance"]
        logger.info(json.dumps(performance["overall"], indent=2))
        metrics = {
            "precision": performance["overall"]["precision"],
            "recall": performance["overall"]["recall"],
            "f1": performance["overall"]["f1"],
            "best_val_loss": artifacts["loss"],
            "behavioral_score": performance["behavioral"]["score"],
            "slices_f1": performance["slices"]["overall"]["f1"],
        }
        mlflow.log_metrics(metrics)

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(
                vars(artifacts["params"]), Path(dp, "params.json"), cls=NumpyEncoder
            )
            utils.save_dict(performance, Path(dp, "performance.json"))
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            artifacts["tokenizer"].save(Path(dp, "tokenizer.json"))
            torch.save(artifacts["model"].state_dict(), Path(dp, "model.pt"))
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
    prediction = predict.predict(texts=[text], artifacts=artifacts)
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


@app.command()
def diff(
    author: str = config.AUTHOR,
    repo: str = config.REPO,
    tag_a: str = "workspace",
    tag_b: str = "",
):  # pragma: no cover, can't be certain what diffs will exist
    """Difference between two release TAGs."""
    # Tag b
    if tag_b == "":
        tags_url = f"https://api.github.com/repos/{author}/{repo}/tags"
        tag_b = utils.load_json_from_url(url=tags_url)[0]["name"]
    logger.info(f"Comparing {tag_a} with {tag_b}:")

    # Params
    params_a = params(author=author, repo=repo, tag=tag_a, verbose=False)
    params_b = params(author=author, repo=repo, tag=tag_b, verbose=False)
    params_diff = utils.dict_diff(
        d_a=params_a, d_b=params_b, d_a_name=tag_a, d_b_name=tag_b
    )
    logger.info(f"Parameter differences: {json.dumps(params_diff, indent=2)}")

    # Performance
    performance_a = performance(author=author, repo=repo, tag=tag_a, verbose=False)
    performance_b = performance(author=author, repo=repo, tag=tag_b, verbose=False)
    performance_diff = utils.dict_diff(
        d_a=performance_a, d_b=performance_b, d_a_name=tag_a, d_b_name=tag_b
    )
    logger.info(f"Performance differences: {json.dumps(performance_diff, indent=2)}")

    return params_diff, performance_diff


# USE RUN_ID HERE!!!
@app.command()
def behavioral_reevaluation(
    model_dir: Path = config.MODEL_DIR,
):  # pragma: no cover, requires changing existing runs
    """Reevaluate existing runs on current behavioral tests in eval.py.
    This is possible since behavioral tests are inputs applied to black box
    models and compared with expected outputs. There is not dependency on
    data or model versions.
    Args:
        model_dir (Path): location of model artifacts.
    Raises:
        ValueError: Run id doesn't exist in experiment.
    """

    # Generate behavioral report
    artifacts = load_artifacts(model_dir=model_dir)
    artifacts["performance"]["behavioral"] = eval.get_behavioral_report(
        artifacts=artifacts
    )
    mlflow.log_metric(
        "behavioral_score", artifacts["performance"]["behavioral"]["score"]
    )

    # Log updated performance
    utils.save_dict(artifacts["performance"], Path(model_dir, "performance.json"))


@app.command()
def get_historical_features():
    """Retrieve historical features for training."""
    # Entities to pull data for (should dynamically read this from somewhere)
    project_ids = [1, 2, 3]
    now = datetime.now()
    timestamps = [datetime(now.year, now.month, now.day)] * len(project_ids)
    entity_df = pd.DataFrame.from_dict(
        {"id": project_ids, "event_timestamp": timestamps}
    )

    # Get historical features
    store = FeatureStore(repo_path=Path(config.BASE_DIR, "features"))
    training_df = store.get_historical_features(
        entity_df=entity_df,
        feature_refs=["project_details:text", "project_details:tags"],
    ).to_df()

    # Store in location for training task to pick up
    print(training_df.head())


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
    label_encoder = data.MultiLabelLabelEncoder.load(
        fp=Path(artifact_uri, "label_encoder.json")
    )
    tokenizer = data.Tokenizer.load(fp=Path(artifact_uri, "tokenizer.json"))
    model_state = torch.load(Path(artifact_uri, "model.pt"), map_location=device)
    performance = utils.load_dict(filepath=Path(artifact_uri, "performance.json"))

    # Initialize model
    model = models.initialize_model(
        params=params, vocab_size=len(tokenizer), num_classes=len(label_encoder)
    )
    model.load_state_dict(model_state)

    return {
        "params": params,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
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
