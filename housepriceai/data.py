# housepriceai/data.py
# Data processing operations.

import itertools
import json
import re
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from config import config
from housepriceai import utils


class houseDataset:
    """Create `torch.utils.data.Dataset` objects to use for
    efficiently feeding data into our models.

    Usage:

    ```python
    # Create dataset
    X, y = data
    dataset = CNNTextDataset(X=X, y=y, max_filter_size=max_filter_size)

    # Create dataloaders
    dataloader = dataset.create_dataloader(batch_size=batch_size)
    ```

    """

    def __init__(self, X, y, max_filter_size):
        self.X = X
        self.y = y
        self.max_filter_size = max_filter_size

    def __len__(self):
        return len(self.y)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index: int) -> List:
        X = self.X[index]
        y = self.y[index]
        return [X, y]

    def collate_fn(self, batch: List) -> Tuple:
        """Processing on a batch. It's used to override the default `collate_fn` in `torch.utils.data.DataLoader`.

        Args:
            batch (List): List of inputs and outputs.

        Returns:
            Processed inputs and outputs.

        """
        # Get inputs
        batch = np.array(batch, dtype=object)
        X = batch[:, 0]
        y = np.stack(batch[:, 1], axis=0)

        # Pad inputs
        X = pad_sequences(sequences=X, max_seq_len=self.max_filter_size)

        # Cast
        X = torch.LongTensor(X.astype(np.int32))
        y = torch.FloatTensor(y.astype(np.int32))

        return X, y


def compute_features(params: Namespace) -> None:
    """Compute features to use for training.

    Args:
        params (Namespace): Input parameters for operations.
    """
    # Set up
    utils.set_seed(seed=params.seed)

    # Load data
    projects_url = (
        "https://raw.githubusercontent.com/HemachandranD/mlops_demo/main/housing.json"
    )
    projects = utils.load_json_from_url(url=projects_url)
    df = pd.DataFrame(projects)

    # Compute features
    # Converting Yes to 1 and No to 0 using map function
    Correspondence = {"yes": 1, "no": 0}

    df["mainroad"] = df["mainroad"].map(Correspondence)
    df["guestroom"] = df["guestroom"].map(Correspondence)
    df["basement"] = df["basement"].map(Correspondence)
    df["hotwaterheating"] = df["hotwaterheating"].map(Correspondence)
    df["airconditioning"] = df["airconditioning"].map(Correspondence)
    df["prefarea"] = df["prefarea"].map(Correspondence)

    # Creating a dummy variable for 'furnishingstatus' or you can say we want to do one-hot encoding on it.
    status = pd.get_dummies(df["furnishingstatus"], drop_first=True)
    df = pd.concat([df, status], axis=1)

    # Rescaling the features using normalize ( ) to all columns using apply function
    df = df.apply(utils.normalize)

    # Save
    features = df.to_dict(orient="records")
    df_dict_fp = Path(config.DATA_DIR, "features.json")
    utils.save_dict(d=features, filepath=df_dict_fp)

    return df, features
