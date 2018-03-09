# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd  # tipp from dimitri (python teacher)
import numpy as np

from sklearn.metrics import mean_squared_error

## TRAIN

# read file
data = pd.read_csv("train.csv")

#turn raw data into matrix
data = data.as_matrix()

# filter matrix
y = data[:, 1]
X = data[:, 2:]

# mean function
y_pred = np.mean(X,1)

# error function
RMSE = mean_squared_error(y, y_pred)**0.5

print(RMSE)

## TEST

# read file
data = pd.read_csv("test.csv")

#turn raw data into matrix
data = data.as_matrix()

# filter matrix
y = data[:, 1]
X = data[:, 2:]

# mean function with nothing learned
y_pred = np.mean(X,1)
col = ["y"]
index = [str(i) for i in range(10000, len(y_pred)+10000)]
#print(len(index))
#print(len(y_pred))
#print(len(col))
# print solution to file
df = pd.DataFrame(y_pred ,col)
df.to_csv("sample.csv")

