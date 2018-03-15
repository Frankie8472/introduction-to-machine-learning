
# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


alphas = np.linspace(90.0, 91.0, 10000*(10**0))
n = 10
tol = 0.0001
rmse = []
mean_rmse = []
best = (0, 0)

# Import data
data = pd.read_csv("train.csv", index_col="Id")

# Convert to matrix
data = data.as_matrix()

# Add functions of x_i
y = data[:, 0]
X = data[:, 1:]
X = np.c_[X, X**2, np.exp(X), np.cos(X), np.ones(np.alen(X))]

# Split into Folds
kf = KFold(n_splits=n, shuffle=False, random_state=None)

# Train
for alpha in alphas:
    clf = Ridge(alpha=alpha,
                fit_intercept=False,
                normalize=False,
                copy_X=True,
                max_iter=1000,
                tol=tol,
                solver="auto",
                random_state=None
                )

    for train_index, test_index in kf.split(data):
        y_train, y_test = data[train_index, 0], data[test_index, 0]
        X_train, X_test = data[train_index, 1:], data[test_index, 1:]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        rmse.append(mean_squared_error(y_test, y_pred) ** 0.5)
    print(str(alpha) + ", " + str(np.mean(rmse)))
    mean_rmse.append(np.mean(rmse))
    rmse = []


# Print solution to file
#result = pd.DataFrame(mean_rmse)
#result.to_csv("sample_franz.csv", index=False, header=False)
for i, j in enumerate(mean_rmse):
    if j == min(mean_rmse):
        print("alpha = " + str(alphas[i]))



# Print solution to file
#result = pd.DataFrame(weights)
#result.to_csv("sample_franz.csv", index=False, header=False)
#print(weights)
