# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd  # tipp from dimitri (python teacher)
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import Lasso

## TRAIN

# read file
raw_data_train = pd.read_csv("train.csv")

# turn raw data into matrix
data_train = raw_data_train.as_matrix()

# filter matrix
y_train = data_train[:, 1]
X_train = data_train[:, 2:]

# Create an SGDClassifier instance which will have methods to do our linear regression fitting by gradient descent
fitter = Lasso()

# Train
fitter.fit(X_train, y_train)

# error function
RMSE_train = mean_squared_error(y_train, fitter.predict(X_train)) ** 0.5
print("RMSE = " + str(RMSE_train))

## TEST

# Read file
raw_data_test = pd.read_csv("test.csv")

# Turn raw data into matrix
data_test = raw_data_test.as_matrix()

# Filter matrix
X_test = data_test[:, 1:]

# Predict
y_pred_test = fitter.predict(X_test)

# Print solution to file
result = pd.DataFrame(data={"Id": raw_data_test["Id"], "y": y_pred_test})
result.to_csv("sample_lasso.csv", index=False)
