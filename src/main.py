from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from tqdm import tqdm

from constants import VIP_1, LABEL_NAME
from src.custom_piplenes import ALL_PAIRS

source_path: Path = Path('./data.xlsx')  # Path("/Users/ysimonov001/Documents/data/fimaLab/lung_cancer")

if __name__ == '__main__':
    common_kwargs = dict(scoring='accuracy',
                         cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42))
    features_names = VIP_1
    data = pd.read_excel(source_path)

    features = data[features_names]
    labels = data[LABEL_NAME]

    tmp_resu = []
    for pair in tqdm(ALL_PAIRS):
        kwargs = pair.copy()
        kwargs.update(common_kwargs)
        gscv = GridSearchCV(**kwargs)
        gscv.fit(features, labels)
        tmp_resu.append(gscv)
    pd.to_pickle(tmp_resu, './grid_search_results.pickle')
