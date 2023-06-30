"""Calc coefs for age deletion"""
from typing import List

import numpy as np
import pandas as pd
from scipy import linalg

from loader import AGE_COLUMN, METABOLITES


def get_coefs(x: pd.Series, y: pd.Series):
    """Calculate k and b for y = kx + b usin scipy.linalg solver"""
    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)
    x = np.hstack([x, np.ones_like(x)])
    return linalg.lstsq(x, y)[0].ravel()


def get_all_coefs(
        data: pd.DataFrame,
        age_column=AGE_COLUMN,
        features: List[str] = METABOLITES,
):
    """Return data without age column"""
    data = data.copy()
    # getting coefs for each group for each feature vs age_column
    coefs = {}
    for feature in features:
        new_coefs = get_coefs(data[age_column], data[feature])
        coefs[feature] = new_coefs
    df_coef = pd.DataFrame(coefs).T
    df_coef.columns = ["k", "b"]
    return df_coef


def rm_age_trend(data: pd.DataFrame, df_coef: pd.DataFrame):
    """Return data without age trend"""
    data = data.copy()
    arguments = np.hstack(
        [data[AGE_COLUMN].values.reshape(-1, 1),
         np.ones_like(data[AGE_COLUMN].values.reshape(-1, 1))]
    )
    wo_trend = {}
    for feature in df_coef.index:
        trend_values = arguments @ df_coef.loc[feature, ["k", "b"]]
        wo_trend[feature] = data[feature] - trend_values
    return pd.DataFrame(wo_trend)
