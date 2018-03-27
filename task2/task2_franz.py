# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
from pandas import read_csv, DataFrame
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV

# Parameter initialization

# Define exercise functions
def read_csv_to_matrix(filename, index_name):
    data = read_csv(filename, index_col=index_name)
    return data.as_matrix(), data.index


def write_to_csv_from_vector(filename, vec):
    return DataFrame(vec).to_csv(filename, index=False, header=["Id", "y"])


def split_into_x_y(data_set):
    y = data_set[:, 0]
    x = data_set[:, 1:]
    return x, y


def root_mean_squared_error(y, y_guess):
    return mean_squared_error(y, y_guess) ** 0.5


# Scorer / Loss function
rmse = make_scorer(root_mean_squared_error, greater_is_better=False)

# Estimator / Predictor
clf = [('KNeighborsClf', KNeighborsClassifier()),
       ('DecisionTreeClf', DecisionTreeClassifier()),
       ('MLPClf', MLPClassifier()),
       ('RandomForestClf', RandomForestClassifier()),
       ('ExtraTreesClf', ExtraTreesClassifier()),
       ('SVC', SVC()),
       ('LinearSVC', LinearSVC()),
       ('BaggingClf', BaggingClassifier())
       ]

# Grid
#gd = GridSearchCV()

for name, clf in clf:

    # Train
    # Get, split and transform dataset
    data_train, index_train = read_csv_to_matrix("train.csv", "Id")
    X_train, y_train = split_into_x_y(data_train)
    clf.fit(X_train, y_train)

    # Test
    # Get, split and transform dataset
    X_test, index_test = read_csv_to_matrix("test.csv", "Id")
    y_pred = np.int8(clf.predict(X_test))

    # Print solution to file
    write_to_csv_from_vector("sample_franz_"+name+".csv", np.c_[index_test, y_pred])

