# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
from pandas import read_csv, DataFrame
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, make_scorer

# Parameter initialization
alphas = np.linspace(14.9, 15.2, num=10)   # np.geomspace(0.001, 30.0, num=1000)
n = 900     # 5, 10, None
fit_intercept = False
normalize = False
gcv_mode = 'auto'     # {None, ‘auto’, ‘svd’, eigen’}

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


# Loss function / scorer
loss_func = make_scorer(score_func=root_mean_squared_error,
                        greater_is_better=False
                        )

# Get, split and transform dataset
data = read_csv_to_matrix("train.csv", "Id")
X, y = split_into_x_y(data)
X = extend_features(X)

# Train
clf = RidgeCV(alphas=alphas,
              fit_intercept=fit_intercept,
              normalize=normalize,
              scoring=loss_func,
              cv=n,
              gcv_mode=gcv_mode
              )

clf.fit(X, y)
best_alpha = clf.alpha_

print("best_alpha = " + str(best_alpha))

weights = clf.coef_

# Print solution to file
write_to_csv_from_vector("sample_franz_ridgecv.csv", weights)
