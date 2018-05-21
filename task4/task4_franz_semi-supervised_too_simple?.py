# train.h5      - the training set
# test.h5       - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

# !-----------------------------!
#  RUN ONLY ON EULER WITH run.sh
# !-----------------------------!

import numpy as np
from pandas import read_hdf, DataFrame
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

if __name__ == "__main__":
    # Parameter initialization
    cores = 48  # Number of cores for parallelization
    message_count = 0  # Bigger = More msgs
    nfolds = [3, 5, 10]  # try out 5 and 10
    iids = [True, False]
    n_components = [None, 0.20, 0.40, 0.60, 0.80, 0.90]
    estimators = {}
    param_grids = {}
    y_pred_list = []
    score_list = []

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


    def parameter_selection(data_train_labeled, X_train_unlabeled, X_test, nfold, iid):
        X_train_labeled, y_train_labeled = split_into_x_y(data_train_labeled)

        mlp_label_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('mlp', MLPClassifier())
        ]

        mlp_full_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('mlp', MLPClassifier())
        ]

        mlp_label_param_grid = {
            'pca__whiten': [True, False],
            'pca__n_components': n_components,
            # use a small layer in the middle (20-70% of the biggest) for countering overfitting ><><
            'mlp__hidden_layer_sizes': [(128,)],  # , (128, 64, 32, 16), (1024, 512, 256, 128)],
            'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'mlp__solver': ['lbfgs', 'sgd', 'adam'],
            'mlp__alpha': np.geomspace(1e-7, 1e2, 10),
            'mlp__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'mlp__max_iter': [1000],
            'mlp__shuffle': [True, False],
            'mlp__random_state': [42],
            'mlp__tol': [1e-6]
        }

        mlp_full_param_grid = {
            'pca__whiten': [True, False],
            'pca__n_components': n_components,
            # use a small layer in the middle (20-70% of the biggest) for countering overfitting ><><
            'mlp__hidden_layer_sizes': [(128,)],  # , (128, 64, 32, 16), (1024, 512, 256, 128)],
            'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'mlp__solver': ['lbfgs', 'sgd', 'adam'],
            'mlp__alpha': np.geomspace(1e-7, 1e2, 10),
            'mlp__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'mlp__max_iter': [10000],
            'mlp__shuffle': [True, False],
            'mlp__random_state': [42],
            'mlp__tol': [1e-6]
        }

        # Scorer / Loss function
        acc = make_scorer(accuracy_score, greater_is_better=True)

        # GridsearchCV initializer
        grid_search_label = GridSearchCV(
            estimator=Pipeline(mlp_label_estimator),
            param_grid=mlp_label_param_grid,
            scoring=acc,
            n_jobs=cores,
            pre_dispatch='2*n_jobs',
            iid=iid,
            cv=stratified_kfold(X_train_labeled, y_train_labeled, n_splits=nfold),
            refit=True,
            verbose=message_count,
            error_score='raise',
            return_train_score=False
        )

        # Train
        grid_search_label.fit(X_train_labeled, y_train_labeled)

        # Label unlabeled
        y_train_unlabeled = grid_search_label.predict(X_train_unlabeled)

        full_X_set = np.r_[X_train_labeled, X_train_unlabeled]
        full_y_set = np.r_[y_train_labeled, y_train_unlabeled]

        # GridsearchCV initializer
        grid_search_full = GridSearchCV(
            estimator=Pipeline(mlp_full_estimator),
            param_grid=mlp_full_param_grid,
            scoring=acc,
            n_jobs=cores,
            pre_dispatch='2*n_jobs',
            iid=iid,
            cv=stratified_kfold(full_X_set, full_y_set, n_splits=nfold),
            refit=True,
            verbose=message_count,
            error_score='raise',
            return_train_score=False
        )

        grid_search_full.fit(full_X_set, full_y_set)

        # Save
        score_list.append(grid_search_full.best_score_)
        y_pred_list.append(grid_search_full.predict(X_test))

        print("======================================================================================")
        print("(iid, nfold): " + "(" + str(iid) + ", " + str(nfold) + ")")
        print("Best score:       " + str(grid_search_full.best_score_))
        print("Best parameters:   ")
        print()
        print(grid_search_full.best_params_)
        print()
        print("======================================================================================")


    # Get, split and transform train dataset
    data_train_labeled, train_labeled_index = read_hdf_to_matrix("train_labeled.h5")
    data_train_unlabeled, train_unlabeled_index = read_hdf_to_matrix("train_unlabeled.h5")
    data_test, test_index = read_hdf_to_matrix("test.h5")

    # Parameter search/evaluation
    for iid in iids:
        for nfold in nfolds:
            parameter_selection(data_train_labeled, data_train_unlabeled, data_test, nfold, iid)

# Get the prediction with the best score
best_score = np.amax(score_list)
best_score_index = score_list.index(best_score)
y_pred_best_score = y_pred_list[best_score_index]

# Print solution to file
write_to_csv_from_vector("sample_franz_semi-supervised.csv", test_index, y_pred_best_score)
