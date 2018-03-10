# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd  # tipp from dimitri (python teacher)
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

## Linear regression

# Load the diabetes dataset
dataset_train = pd.read_csv("train.csv")
raw_dataset_test = pd.read_csv("test.csv")

dataset_train = dataset_train.as_matrix()
dataset_test = raw_dataset_test.as_matrix()

dataset_train_X = dataset_train[:, 2:]
dataset_train_y = dataset_train[:, 1]
dataset_test_X = dataset_test[:, 1:]

# Use only one feature
# How?

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(dataset_train_X, dataset_train_y)

# error function
RMSE_train = mean_squared_error(dataset_train_y, regr.predict(dataset_train_X)) ** 0.5
print("RMSE = " + str(RMSE_train))

# Make predictions using the testing set
dataset_y_pred = regr.predict(dataset_test_X)

# Print solution to file
result = pd.DataFrame(data={"Id": raw_dataset_test["Id"], "y": dataset_y_pred})
result.to_csv("sample_linear_regression.csv", index=False)
