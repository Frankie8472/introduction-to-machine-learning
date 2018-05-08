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
data_train_labeled, index_train_labeled = read_hdf_to_matrix("train_labeled.h5")
X_train_labeled, y_train_labeled = split_into_x_y(data_train_labeled)

ctr = col.Counter(y_train_labeled)
occurrences = list(ctr.values())
labeled_list = []
for key, value in ctr.items():
    labeled_list.append((value, np.int(key)))
labeled_list.sort(reverse=True)

print("====================================================================================")

for value, key in labeled_list:
    print(" " + str(key), end="       ")

print()

for value, key in labeled_list:
    print(value, end="      ")

print()
print("====================================================================================")

if np.std(occurrences) < 0.2*np.mean(occurrences):
    print("Feature re-sampling is NOT recommended")
else:
    print("Feature re-sampling IS recommended")

print("====================================================================================")

print("Shape of labeled_data:   " + str(np.shape(X_train_labeled)))
data_train_unlabeled, data_train_unlabeled_index = read_hdf_to_matrix("train_unlabeled.h5")
print("Shape of unlabeled_data: " + str(np.shape(data_train_unlabeled)))
data_test, data_test_index = read_hdf_to_matrix("test.h5")
print("Shape of test_data:      " + str(np.shape(data_test)))

print("====================================================================================")














