# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
from pandas import read_csv, DataFrame
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Parameter initialization
alphas = np.linspace(14, 15.5, num=100)  # np.geomspace(0.00001, 100.0, num=10000)
n = 900     # 5, 10, 900
fit_intercept = False
normalize = False
max_iter = 1000
tol = 0.0001
solver = 'auto'   # ‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’
random_state = 1
rmse = []
mean_rmse = []
mean_std = []


# Define exercise functions
def read_csv_to_matrix(filename, index_name):
    return read_csv(filename, index_col=index_name).as_matrix()


def write_to_csv_from_vector(filename, vec):
    return DataFrame(vec).to_csv(filename, index=False, header=False)


def split_into_x_y(data_set):
    y = data_set[:, 0]
    x = data_set[:, 1:]
    return x, y


def root_mean_squared_error(y, y_guess):
    return mean_squared_error(y, y_guess) ** 0.5


def square_element_wise(matrix):
    return matrix ** 2


def exp_element_wise(matrix):
    return np.exp(matrix)


def cos_element_wise(matrix):
    return np.cos(matrix)


def ones_vector_same_length(matrix):
    return np.ones(np.alen(matrix))


def extend_features(features):
    return np.c_[features,
                 square_element_wise(features),
                 exp_element_wise(features),
                 cos_element_wise(features),
                 ones_vector_same_length(features)]


# Get, split and transform dataset
data = read_csv_to_matrix("train.csv", "Id")
X, y = split_into_x_y(data)
X = extend_features(X)
kf = KFold(n_splits=n, shuffle=False, random_state=None)

# Train
for alpha in alphas:
    clf = Ridge(alpha=alpha,
                fit_intercept=fit_intercept,
                normalize=normalize,
                max_iter=max_iter,
                tol=tol,
                solver=solver,
                random_state=random_state
                )

    for train_index, test_index in kf.split(y):
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X[train_index, :], X[test_index, :]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        rmse.append(root_mean_squared_error(y_test, y_pred))

    mean_rmse.append((np.mean(rmse)))
    mean_std.append(np.std(rmse))
    rmse = []

combi = (np.array(mean_rmse) + np.array(mean_std)).tolist()
min_rmse = min(mean_rmse)
min_std = min(mean_std)
min_combi = min(combi)
idx_combi = combi.index(min_combi)
idx_rmse = mean_rmse.index(min_rmse)
idx_std = mean_std.index(min_std)
best_alpha = alphas[idx_combi]

print("min_rmse = " + str(min_rmse) + ", " + str(mean_std[idx_rmse]) + ", " + str(idx_rmse))
print("min_std = " + str(min_std) + ", " + str(idx_std))
print("min_combi = " + str(min_combi) + ", " + str(mean_std[idx_combi]) + ", " + str(idx_combi))
print("alpha of min_combi = " + str(alphas[idx_rmse]))

clf = Ridge(alpha=best_alpha,
            fit_intercept=fit_intercept,
            normalize=normalize,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            random_state=random_state
            )

clf.fit(X_train, y_train)
weights = clf.coef_

# Print solution to file
write_to_csv_from_vector("sample_franz.csv", weights)
