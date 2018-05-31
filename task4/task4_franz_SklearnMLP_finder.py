# train_labeled.h5      - the labeled training set
# train_unlabeled.h5    - the unlabeled training set
# test.h5               - the test set (make predictions based on this file)
# sample.csv            - a sample submission file in the correct format

# !-----------------------------!
#  RUN ONLY ON EULER WITH run.sh
# !-----------------------------!

import numpy as np
from multiprocessing import cpu_count
from gc import collect
from pandas import read_hdf, DataFrame
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Parameter initialization
cores = 48  # Number of cores for parallelization (3, 4, 48)
message_count = 0  # Bigger = More msgs
nfold = 10
n_components = [None, 90, 70, 50, 40, 30]
'''
mlp = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation='relu',
            solver='lbfgs',
            alpha=13.0,
            learning_rate='constant',
            max_iter=200,
            shuffle=False,
            tol=1e-5
        )

'''

# Define exercise functions
def read_hdf_to_matrix(filename):
    data = read_hdf("input/" + filename)
    return data.as_matrix(), data.index


def write_to_csv_from_vector(filename, index_col, vec):
    return DataFrame(np.c_[index_col, vec]).to_csv("output/" + filename, index=False, header=["Id", "y"])


def split_into_x_y(data_set):
    y = data_set[:, 0]
    x = data_set[:, 1:]
    return x, y


def stratified_kfold(X, y, *args, **kwargs):
    skf = StratifiedKFold(*args, **kwargs)
    for train, test in skf.split(X, y):
        yield train, test


def parameter_selection(data_labeled):
    X_train_labeled, y_train_labeled = split_into_x_y(data_labeled)

    mlp_estimator = [
        ('pca', PCA()),
        ('ss', StandardScaler()),
        ('mlp', MLPClassifier())
    ]

    mlp_param_grid = {
        'pca__whiten': [True, False],
        'pca__n_components': n_components,
        # use a small layer in the middle (20-70% of the biggest) for countering overfitting ><><
        'mlp__hidden_layer_sizes': [(128,), (400,), (400, 400, 10)],  # , (128, 64, 32, 16), (1024, 512, 256, 128)],
        'mlp__activation': ['tanh', 'relu'],
        'mlp__solver': ['lbfgs'],
        'mlp__alpha': np.linspace(0, 20, 10),
        'mlp__learning_rate': ['constant'],
        'mlp__max_iter': [1000],
        'mlp__shuffle': [False],
        'mlp__random_state': [42],
        'mlp__tol': [1e-5]
    }

    # Scorer / Loss function
    acc = make_scorer(accuracy_score, greater_is_better=True)

    # GridsearchCV initializer
    grid_search = GridSearchCV(
        estimator=Pipeline(mlp_estimator),
        param_grid=mlp_param_grid,
        scoring=acc,
        n_jobs=cores,
        pre_dispatch='2*n_jobs',
        iid=False,
        cv=stratified_kfold(X_train_labeled, y_train_labeled, n_splits=nfold),
        refit=True,
        verbose=message_count,
        error_score='raise',
        return_train_score=False
    )

    # Train
    grid_search.fit(X_train_labeled, y_train_labeled)

    print("======================================================================================")
    print("Best score:       " + str(grid_search.best_score_))
    print("Best parameters:   ")
    print("")
    print(grid_search.best_params_)
    print("")
    print("======================================================================================")


if __name__ == "__main__":
    # Get, split and transform train dataset
    data_train_labeled, data_train_labeled_index = read_hdf_to_matrix("train_labeled.h5")

    # Parameter search/evaluation
    parameter_selection(data_train_labeled)
