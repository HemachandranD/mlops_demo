# housepriceai/models.py
# Model architectures.

import math
from argparse import Namespace
from typing import List

import statsmodels.api as sm


class SM:
    def __init__(self, X_train, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train
        """A stats model api for performing the regression

        Args:
            """
        lm = sm.OLS(y_train, X_train)
        z = lm.fit()
        return z


def initialize_model(params: Namespace, X_train, y_train):
    """Initialize a model using parameters (converted to appropriate data types).

    Args:
        params (Namespace): Parameters for data processing and training.
        vocab_size (int): Size of the vocabulary.
        num_classes (int): Number on unique classes.
        device (torch.device): Device to run model on. Defaults to CPU.

    Returns:
        Initialize torch model instance.
    """
    # Initialize model
    model = SM(X_train, y_train)
    return model
