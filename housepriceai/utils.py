# housepriceai/utils.py
# Utility functions.

import json
import numbers
import random

from pandas.core.frame import DataFrame
import urllib2
from typing import Dict, List
from urllib.request import urlopen

import numpy as np
import pandas as pd


def load_csv_from_url(url: str) -> Dict:
    """Load CSV data from a URL.
    Args:
        url (str): URL of the data source.
    Returns:
        A dictionary with the loaded CSV data.
    """
    data = urllib2.urlopen(url)
    return data


def load_json_from_url(url: str) -> Dict:
    """Load JSON data from a URL.
    Args:
        url (str): URL of the data source.
    Returns:
        A dictionary with the loaded JSON data.
    """
    data = json.loads(urlopen(url).read())
    return data


def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.
    Args:
        filepath (str): JSON's filepath.
    Returns:
        A dictionary with the data loaded.
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d


def normalize(x: DataFrame) -> DataFrame:
    """Load a DataFrame.
    Args:
        x (DataFrame): df DataFrame.
    Returns:
        A min-max scaled DataFrame.
    """
    return (x - min(x)) / (max(x) - min(x))


def save_dict(d: Dict, filepath: str) -> None:
    """Save a dictionary to a specific location.
    Warning:
        This will overwrite any existing file at `filepath`.
    Args:
        d (Dict): dictionary to save.
        filepath (str): location to save the dictionary to as a JSON file.
      """
    with open(filepath, "w") as fp:
        json.dump(d, p=fp)


def set_seed(seed: int = 1234) -> None:
    """Set seed for reproducibility.
    Args:
        seed (int, optional): number to use as the seed. Defaults to 1234.
    """
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
