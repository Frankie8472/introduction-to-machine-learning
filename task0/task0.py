# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd  # tipp from dimitri (python teacher)
import numpy as np

from sklearn.metrics import mean_squared_error

## TRAIN

# read file
data_train = pd.read_csv("train.csv")

#turn raw data into matrix
data_train = data_train.as_matrix()

# filter matrix
y = data_train[:, 1]
X = data_train[:, 2:]

# mean function
y_pred = np.mean(X,1)

# error function
RMSE = mean_squared_error(y, y_pred)**0.5

## TEST

# read file
data_test = pd.read_csv("test.csv")

#turn raw data into matrix
data_test = data_test.as_matrix()

# filter matrix
X = data_test[:, 1:]

# mean function with nothing learned
y_pred = np.mean(X,1)

# print solution to file
result = pd.DataFrame(data={"Id": data_test[:, 0], "y": y_pred})
result.to_csv("sample.csv", index=False)

