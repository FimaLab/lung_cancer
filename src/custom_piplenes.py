import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# STEPS
SIMPLE_IMPUTER = SimpleImputer(strategy='median')
STANDARD_SCALER = StandardScaler()


LOG_PIPE = Pipeline([
    ('simple_imputer', SIMPLE_IMPUTER),
    ('standard_scaler', STANDARD_SCALER),
    ('logistic_regression', LogisticRegression())
])
SVM_PIPE = Pipeline([
    ('simple_imputer', SIMPLE_IMPUTER),
    ('standard_scaler', STANDARD_SCALER),
    ('svc', SVC())
])
RF_PIPE = Pipeline([
    ('simple_imputer', SIMPLE_IMPUTER),
    ('random_forest_classifier', RandomForestClassifier())
])
GB_PIPE = Pipeline([
    ('simple_imputer', SIMPLE_IMPUTER),
    ('gradient_boosting_classifier', GradientBoostingClassifier())
])

LOG_PARAMETRS = {
    'logistic_regression__penalty': ['l1', 'l2'],
    'logistic_regression__C': np.logspace(-3, 1, 20),
    'logistic_regression__solver': ['newton-cg',  'lbfgs', 'liblinear'],
}
SVM_PARAMETRS = {
    'svc__gamma': [1, 0.1, 0.01, 0.001],
    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'svc__C': np.logspace(-3,3,7)
}
RF_PARAMETRS = {
    'random_forest_classifier__n_estimators': [100],
    'random_forest_classifier__max_features': ['sqrt'],
    'random_forest_classifier__max_depth': [3],
    'random_forest_classifier__criterion': ['gini']
}
GB_PARAMETRS = {
    'gradient_boosting_classifier__loss': ["deviance", "exponential"],
    'gradient_boosting_classifier__max_features': ["log2", "sqrt"],
    'gradient_boosting_classifier__learning_rate': [0.1, 1, 0.5],  # so called `eta` value
    'gradient_boosting_classifier__max_depth': [5],
    'gradient_boosting_classifier__min_samples_leaf': [1, 2, 4],
    'gradient_boosting_classifier__subsample': [0.1, 0.3],
    'gradient_boosting_classifier__n_estimators': [20, 30, 60, 100]
}
ALL_PAIRS = (
    {'estimator': LOG_PIPE, 'param_grid': LOG_PARAMETRS},
    {'estimator': SVM_PIPE, 'param_grid': SVM_PARAMETRS},
    {'estimator': RF_PIPE, 'param_grid': RF_PARAMETRS},
    {'estimator': GB_PIPE, 'param_grid': GB_PARAMETRS}
)
