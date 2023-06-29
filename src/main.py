from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from tqdm import tqdm

from constants import VIP_1, LABEL_NAME
from src.custom_piplenes import ALL_PAIRS

# config part
source_path: Path = Path('/Users/ksu_/Documents/notebooks/CVD/CVDC/Файлы для ввода/IHD_AMI.xlsx')
features_names = None
result_folder = Path(".")

if __name__ == '__main__':
    data = pd.read_excel(source_path)
    common_kwargs = dict(scoring='accuracy',
                         cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42))
    if features_names is None:
        features_names = data.columns.drop(LABEL_NAME)
    else:
        features_names
    print(data.columns)

    features = data[features_names]
    labels = data[LABEL_NAME]

    tmp_resu = []
    for pair in ALL_PAIRS:
        kwargs = pair.copy()
        kwargs.update(common_kwargs)
        gscv = GridSearchCV(**kwargs)
        gscv.fit(features, labels)
        tmp_resu.append(gscv)
    pd.to_pickle(tmp_resu, result_folder / 'grid_search_results.pickle')
