
# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Parameter initialization
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
n = 10
rmse = []
mean_rmse = []

# Import data and split
data = pd.read_csv("train.csv", index_col="Id")
data = data.as_matrix()
kf = KFold(n_splits=n, shuffle=False, random_state=None)

# Train
for alpha in alphas:
    clf = Ridge(alpha=alpha, fit_intercept=False)   # Assume the data is centered

    for train_index, test_index in kf.split(data):
        y_train, y_test = data[train_index, 0], data[test_index, 0]
        X_train, X_test = data[train_index, 1:], data[test_index, 1:]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        rmse.append(mean_squared_error(y_test, y_pred) ** 0.5)

    mean_rmse.append(np.mean(rmse))
    rmse = []

# Print solution to file
result = pd.DataFrame(mean_rmse)
result.to_csv("sample_franz.csv", index=False, header=False)

print(mean_rmse)
