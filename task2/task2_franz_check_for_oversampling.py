# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
import collections as col
from pandas import read_csv, DataFrame
from sklearn.metrics import accuracy_score, make_scorer


# Define exercise functions
def read_csv_to_matrix(filename, index_name):
    data = read_csv(filename, index_col=index_name)
    return data.as_matrix(), data.index


def write_to_csv_from_vector(filename, vec):
    return DataFrame(vec).to_csv(filename, index=False, header=["Id", "y"])


def split_into_x_y(data_set):
    y = data_set[:, 0]
    x = data_set[:, 1:]
    return x, y


# Scorer / Loss function
acc = make_scorer(accuracy_score, greater_is_better=True)


# Get, split and transform train dataset
data_train, index_train = read_csv_to_matrix("train.csv", "Id")
X_train, y_train = split_into_x_y(data_train)

ctr = col.Counter(y_train)
occ = list(ctr.values())

if (np.std(occ) < 0.1*np.mean(occ)):
    print("Features DO NOT require resampling")
else:
    print("Features DO require resampling")
