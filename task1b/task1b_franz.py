# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
from pandas import read_csv
from pandas import DataFrame
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import make_scorer, mean_squared_error

# Define parameter specs
alphas = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


# Define exercise functions
def read_csv_to_matrix(filename, index_name):
    return read_csv(filename, index_col=index_name).as_matrix()


def write_to_csv_from_vector(filename, vec):
    return DataFrame(vec).to_csv(filename, index=False, header=False)


def split_into_x_y(data_set):
    y = data_set[:, 0]
    x = data_set[:, 1:]
    return x, y


def root_mean_squared_error(y, y_pred):
    return mean_squared_error(y, y_pred) ** 0.5


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
X_train, y_train = split_into_x_y(data)
X_train = extend_features(X_train)

# Loss function
loss_func = make_scorer(root_mean_squared_error, greater_is_better=False)

# Regressor
reg = RidgeCV(alphas=alphas,
              fit_intercept=True,
              scoring=loss_func,
              cv=None)

# Train with cv
reg.fit(X_train, y_train)

# Get ideal parameters
best_alpha = reg.alpha_
print(best_alpha)

clf = Ridge(alpha=best_alpha,
            fit_intercept=True,
            )
clf.fit(X_train, y_train)

# Calculate weights with the ideal parameters
weights = clf.coef_

# Write result in file
write_to_csv_from_vector("sample_franz.csv", weights)
