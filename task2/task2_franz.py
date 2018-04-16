# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
from pandas import read_csv, DataFrame
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.decomposition import PCA

# Parameter initialization
cores = 4
message_count = 10


# Define exercise functions
def read_csv_to_matrix(filename, index_name):
    data = read_csv("input/" + filename, index_col=index_name)
    return data.as_matrix(), data.index


def write_to_csv_from_vector(filename, vec):
    return DataFrame(vec).to_csv("output/" + filename, index=False, header=["Id", "y"])


def split_into_x_y(data_set):
    y = data_set[:, 0]
    x = data_set[:, 1:]
    return x, y


def param_selection(nfold, iid, X, y, name):
    param_grid = {
        'penalty': ['l1', 'l2'],
        'loss': ['hinge', 'squared_hinge'],
        'dual': [True, False],
        'tol': np.geomspace(10**-8, 1, 9),
        'C': np.geomspace(10**-10, 10**0, 11),
        'multi_class': ['crammer_singer'],
        'fit_intercept': [True, False],
        'max_iter': [1000]
    }

    # Scorer / Loss function
    acc = make_scorer(accuracy_score, greater_is_better=True)

    # Estimator / Predictor
    clf = LinearSVC(
        penalty='l2',
        loss='squared_hinge',
        dual=True,
        tol=10**-4,
        C=1.0,
        multi_class='ovr',
        fit_intercept=True,
        intercept_scaling=1,
        max_iter=1000
    )

    # GridsearchCV
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring=acc,
        n_jobs=cores,
        pre_dispatch='2*n_jobs',
        iid=iid,
        cv=nfold,
        refit=True,
        verbose=message_count,
        error_score='raise',
        return_train_score=False,
    )

    # Train
    grid_search.fit(X, y)

    print("===================================================")
    print("For (iid, nfold, set): " + "(" + str(iid) + ", " + str(nfold) + ", " + name + ")")
    print("Best score:       " + str(grid_search.best_score_))
    print("Best score refit: " + str(grid_search.score(X, y)))
    print("Best parameter:   ")
    print("")
    print(grid_search.best_params_)
    print("")
    print("===================================================")


# Get, split and transform train dataset
data_train, index_train = read_csv_to_matrix("train.csv", "Id")
X, y = split_into_x_y(data_train)

# Preprocessing
data_sets = []

scalerSTD = StandardScaler(
    copy=True,
    with_mean=True,
    with_std=True
)

scalerPCA = PCA(
    # n_components=,
    copy=True,
    whiten=False,   # Try with True
    svd_solver='auto',
    tol=.0,
    iterated_power='auto',
)

scalerPoly = PolynomialFeatures(
    degree=3,
    interaction_only=False,
    include_bias=True   # Try False
)

scalerMinMax = MinMaxScaler(
    feature_range=(0, 1),
    copy=True
)

scalerSTD.fit(X, y)
scalerPCA.fit(X, y)
scalerPoly.fit(X, y)
scalerMinMax.fit(X, y)

data_sets.append(('STD', scalerSTD.transform(X), y))
data_sets.append(('PCA', scalerPCA.transform(X), y))
data_sets.append(('Poly', scalerPoly.transform(X), y))
data_sets.append(('MinMax', scalerMinMax.transform(X), y))

# Find best parameters
for name, X, y in data_sets:
    for iid in [True, False]:
        for nfold in [5]:     # for leave-one-out: np.size(X_train, 0)
            param_selection(nfold, iid, X, y, name)

# Get, split and transform test dataset
# X_test, index_test = read_csv_to_matrix("test.csv", "Id")

# Predict
# y_pred = np.int8(clf_searcher.predict(X_test))

# Print solution to file
# write_to_csv_from_vector("sample_franz_svn.csv", np.c_[index_test, y_pred])
