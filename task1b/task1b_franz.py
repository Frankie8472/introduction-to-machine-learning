
# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
n = 10
seed = None     # Integer for same output
rmse = []
mean_rmse = []

# Import data
data = pd.read_csv("train.csv", index_col="Id")

# Convert to matrix
data = data.as_matrix()

# Add functions of x_i
X = data[:, 1:]
data = np.c_[data, X**2, np.exp(X), np.cos(X), np.ones(np.alen(X))]
X = data[:, 1:]
y = data[:, 0]

# Linear regression
clf = LinearRegression()
clf.fit(X, y)
w = clf.coef_
# Print solution to file
result = pd.DataFrame(w)
result.to_csv("sample_franz.csv", index=False, header=False)
