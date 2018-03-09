# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd  # tipp from dimitri (python teacher)
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

## TRAIN

# read file
data_train = pd.read_csv("train.csv")

# turn raw data into matrix
data_train = data_train.as_matrix()

# filter matrix
y = data_train[:, 1]
X = data_train[:, 2:]

# mean function
y_pred = np.mean(X, 1)

# error function
RMSE_train = mean_squared_error(y, y_pred) ** 0.5

## TEST

# read file
raw_data_test = pd.read_csv("test.csv")

# turn raw data into matrix
data_test = raw_data_test.as_matrix()

# filter matrix
X_test = data_test[:, 1:]

# mean function with nothing learned
y_pred_test = np.mean(X_test, 1)

# print solution to file
result = pd.DataFrame(data={"Id": np.squeeze(raw_data_test.as_matrix(columns=["Id"])), "y": y_pred_test})
result.to_csv("sample.csv", index=False)

## FANCY SHIT

# Load the diabetes dataset
dataset_train = pd.read_csv("train.csv")
raw_dataset_test = pd.read_csv("test.csv")

dataset_train = dataset_train.as_matrix()
dataset_test = raw_dataset_test.as_matrix()

dataset_train_X = dataset_train[:, 2:]
dataset_train_y = dataset_train[:, 1]
dataset_test_X = dataset_test[:, 1:]

# Use only one feature
# how?

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(dataset_train_X, dataset_train_y)

# Make predictions using the testing set
dataset_y_pred = regr.predict(dataset_test_X)

# print solution to file
result = pd.DataFrame(data={"Id": np.squeeze(raw_data_test.as_matrix(columns=["Id"])), "y": dataset_y_pred})
result.to_csv("sample_linear_regression.csv", index=False)
