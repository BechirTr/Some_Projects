import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit
import numpy as np

problem_title = 'Salary prediction'
_target_column_name = 'SalaryUSD'


# A value which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()


# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.RMSE(name='rmse', precision=3),
]

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=57)
    return cv.split(X, y)


# READ DATA

def _read_data(path, df_filename):
    df = pd.read_csv(os.path.join(path, 'data', df_filename), index_col=0)
    y = df[_target_column_name]
    X = df.drop(_target_column_name, axis=1)
    return X, np.log(1+y.values)

def get_train_data(path='.'):
    df_filename = 'train.csv'
    
    return _read_data(path, df_filename)


def get_test_data(path='.'):
    df_filename = 'test.csv'
    return _read_data(path, df_filename)
