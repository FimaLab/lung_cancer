from pathlib import Path

import pandas as pd

from constants import LABEL_NAME, AGE_COLUMN, METABOLITES, GROUP_COLUMN


def get_data(path: Path) -> pd.DataFrame:
    """
    Load data from excel file
    and fill missing values of metabolites with median values
    Parameters
    ----------
    path: Path
        Path to excel file

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_excel(path, index_col="Number")
    df = df[df[LABEL_NAME] != "AMI"]
    df = df[df[AGE_COLUMN].notnull()]
    med_values = df.groupby(LABEL_NAME)[METABOLITES].median()
    df[GROUP_COLUMN] = df[AGE_COLUMN] // 5
    return df.set_index(LABEL_NAME).fillna(med_values).reset_index()
