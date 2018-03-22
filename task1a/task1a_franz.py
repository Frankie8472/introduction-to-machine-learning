
# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd
import numpy as np
from pandas import read_csv, DataFrame
from numpy import mean
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, make_scorer

# Parameter initialization
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
n = 10
rmse = []
mean_rmse = []


# Define exercise functions
def read_csv_to_matrix(filename, index_name):
    return read_csv(filename, index_col=index_name).as_matrix()


def write_to_csv_from_vector(filename, vec):
    return DataFrame(vec).to_csv(filename, index=False, header=False)


def split_into_x_y(data_set):
    y = data_set[:, 0]
    x = data_set[:, 1:]
    return x, y


def root_mean_squared_error(y, y_pred):
    return mean_squared_error(y, y_pred) ** 0.5


# Scorer
loss_func = make_scorer(root_mean_squared_error, greater_is_better=False)


# Import data and split
data = read_csv_to_matrix("train.csv", "Id")
kf = KFold(n_splits=n, shuffle=False, random_state=None)

# Train
for alpha in alphas:
    clf = Ridge(alpha=alpha, fit_intercept=False)   # Assume the data is centered

    for train_index, test_index in kf.split(data):
        y_train, y_test = data[train_index, 0], data[test_index, 0]
        X_train, X_test = data[train_index, 1:], data[test_index, 1:]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        rmse.append(root_mean_squared_error(y_test, y_pred))

    mean_rmse.append(mean(rmse))
    rmse = []

# Print solution to file
write_to_csv_from_vector("sample_franz.csv", mean_rmse)
