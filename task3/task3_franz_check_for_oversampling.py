# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
import collections as col
from pandas import read_hdf


# Define exercise functions
def read_hdf_to_matrix(filename):
    data = read_hdf("input/" + filename)
    return data.as_matrix(), data.index


def split_into_x_y(data_set):
    y = data_set[:, 0]
    x = data_set[:, 1:]
    return x, y


# Get, split and transform train dataset
data_train, index_train = read_hdf_to_matrix("train.h5")
X_train, y_train = split_into_x_y(data_train)

ctr = col.Counter(y_train)
occurrences = list(ctr.values())
print("=======================================================================")
print(ctr)
print("=======================================================================")
if np.std(occurrences) < 0.2*np.mean(occurrences):
    print("Features DO NOT require re-sampling")
else:
    print("Features DO require re-sampling")
print("=======================================================================")
