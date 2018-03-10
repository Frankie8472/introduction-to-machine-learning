# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd  # tipp from dimitri (python teacher)
import numpy as np
from sklearn.metrics import mean_squared_error


precision = np.float128

## TRAIN

# read file
raw_data_train = pd.read_csv("train.csv")

# turn raw data into matrix
data_train = raw_data_train.as_matrix()

# filter matrix
y_train = np.asarray(data_train[:, 1], dtype=precision)
X_train = np.asarray(data_train[:, 2:], dtype=precision)

# Sort before adding (floating point), makes it actually worse...
X_train = np.sort(X_train, 1)

# mean function
y_pred = np.mean(X_train, 1)

# Error function
RMSE_train = mean_squared_error(y_train, y_pred) ** 0.5
print("RMSE = " + str(RMSE_train))

## TEST

# Read file
raw_data_test = pd.read_csv("test.csv")

# Turn raw data into matrix
data_test = raw_data_test.as_matrix()

# Filter matrix
X_test = np.asarray(data_test[:, 1:], dtype=precision)

# Sort before adding (floating point), makes it actually worse...
X_test = np.sort(X_test, 1)

# Mean function with nothing learned
y_pred_test = np.mean(X_test, 1)

# Print solution to file
result = pd.DataFrame(data={"Id": raw_data_test["Id"], "y": y_pred_test})
result.to_csv("sample_max_precision.csv", index=False)
