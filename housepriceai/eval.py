# housepriceai/eval.py
# Evaluation components.

import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from housepriceai import data, predict, train


def regression_metrics(y_test, y_pred):
    """Load JSON data from a URL.
    Args:
        y_test: Test Data of Target Variable.
        y_pred: Predicted Target Data by the model.
    Returns:
        A Regression metrics of MSE, RMSE, MAE.
    """
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MAE = mean_absolute_error(y_test, y_pred)
    return MSE, RMSE, MAE
