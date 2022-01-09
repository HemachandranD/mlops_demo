# housepriceai/train.py
# Training operations.

import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from config import config
from config.config import logger
from housepriceai import data, eval, models, utils


class Trainer:
    def __init__(self, X_train, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train
        """A stats model api for performing the regression

        Args:
            """

        return t


def train(params: Namespace) -> Dict:
    """Operations for training.
    Args:
        params (Namespace): Input parameters for operations.
        trial (optuna.trial._trial.Trail, optional): Optuna optimization trial. Defaults to None.
    Returns:
        Artifacts to save and load for later.
    """

    # Load features
    features_fp = Path(config.DATA_DIR, "features.json")
    features = utils.load_dict(filepath=features_fp)
    df = pd.DataFrame(features)

    # Putting feature variable to X
    target_col = "price"
    X = df.drop(
        [
            "bedrooms",
            "bbratio",
            "areaperbedroom",
            "furnishingstatus" "semi-furnished",
            "basement",
            target_col,
        ],
        axis=1,
    )

    # Putting response variable to y ie. price
    y = df.loc[:, target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, params.test_size, params.random_state
    )

    # Initialize model
    model = models.initialize_model(params=params, X_train=X_train, y_train=y_train)

    # Train model
    logger.info(f"Parameters: {json.dumps(params.__dict__)}")

    # Evaluate model
    artifacts = {
        "params": params,
        "model": model,
    }

    return artifacts

