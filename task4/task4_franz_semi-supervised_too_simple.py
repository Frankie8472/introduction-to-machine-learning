# train.h5      - the training set
# test.h5       - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

# !-----------------------------!
#  RUN ONLY ON EULER WITH run.sh
# !-----------------------------!

import numpy as np
import collections as col
from pandas import read_hdf, DataFrame
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC, SVC

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
y_pred_best_score = []
techs = ['mlp', 'lsvc', 'svc', 'knc', 'rfc', 'etc', 'gbc']


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


def parameter_selection(data_train_labeled, X_train_unlabeled, X_test, nfold, iid, tech):
    X_train_labeled, y_train_labeled = split_into_x_y(data_train_labeled)

    mlp_estimator = [
        ('ss', StandardScaler()),
        ('pf', PolynomialFeatures()),
        ('pca', PCA()),
        ('mlp', MLPClassifier())
    ]

    lsvc_estimator = [
        ('ss', StandardScaler()),
        ('pf', PolynomialFeatures()),
        ('pca', PCA()),
        ('lsvc', LinearSVC())
    ]

    svc_estimator = [
        ('ss', StandardScaler()),
        ('pf', PolynomialFeatures()),
        ('pca', PCA()),
        ('svc', SVC())
    ]

    knc_estimator = [
        ('ss', StandardScaler()),
        ('pf', PolynomialFeatures()),
        ('pca', PCA()),
        ('knc', KNeighborsClassifier())
    ]

    rfc_estimator = [
        ('ss', StandardScaler()),
        ('pf', PolynomialFeatures()),
        ('pca', PCA()),
        ('rfc', RandomForestClassifier())
    ]

    etc_estimator = [
        ('ss', StandardScaler()),
        ('pf', PolynomialFeatures()),
        ('pca', PCA()),
        ('etc', ExtraTreesClassifier())
    ]

    gbc_estimator = [
        ('ss', StandardScaler()),
        ('pf', PolynomialFeatures()),
        ('pca', PCA()),
        ('gbc', GradientBoostingClassifier())
    ]

    mlp_full_estimator = [
        ('ss', StandardScaler()),
        ('pf', PolynomialFeatures()),
        ('pca', PCA()),
        ('mlp', MLPClassifier())
    ]

    mlp_param_grid = {
        'pf__degree': [1, 2, 3],
        'pca__whiten': [True, False],
        'pca__n_components': n_components,
        # use a small layer in the middle (20-70% of the biggest) for countering overfitting ><><
        # 'mlp__hidden_layer_sizes': [(128,)],   # , (128, 64, 32, 16), (1024, 512, 256, 128)],
        # 'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
        # 'mlp__solver': ['lbfgs', 'sgd', 'adam'],
        # 'mlp__alpha': np.linspace(0, 20, 5),
        # 'mlp__learning_rate': ['constant', 'invscaling', 'adaptive'],
        # 'mlp__max_iter': [1000],
        # 'mlp__shuffle': [False],
        # 'mlp__random_state': [42],
        # 'mlp__tol': [1e-5]
    }

    lsvc_param_grid = {
        'pf__degree': [1, 2, 3],
        'pca__whiten': [True, False],
        'pca__n_components': n_components,
        # 'lsvc__penalty': ['l2'],
        # 'lsvc__loss': ['hinge', 'squared_hinge'],
        # 'lsvc__dual': [True],   #False or hinge and l1
        # 'lsvc__tol': [1e-6],
        # 'lsvc__C': np.linspace(4, 20, 9),
        # 'lsvc__multi_class': ['ovr', 'crammer_singer'],
        # 'lsvc__fit_intercept': [True, False],
        # 'lsvc__max_iter': [10000]
    }

    svc_param_grid = {
        'pf__degree': [1, 2, 3],
        'pca__whiten': [True, False],
        'pca__n_components': n_components,
        # 'svc__C': np.geomspace(1e-7, 1e2, 10),
        # 'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        # 'svc__degree': [0, 1, 2, 3],    # mby try out 4?
        # #'svc__gamma': ['auto'],
        # #'svc__coef0': [0.0],
        # #'svc__probability': [True, False],
        # #'svc__shrinking': [True, False],
        # #'svc__max_iter': [-1],
        # 'svc__decision_function_shape': ['ovo', 'ovr']
    }

    knc_param_grid = {
        'pf__degree': [1, 2, 3],
        'pca__whiten': [True, False],
        'pca__n_components': n_components,
        # 'knc__n_neighbors': [9, 10],
        # 'knc__weights': ['uniform', 'distance'],
        # 'knc__algorithm': ['ball_tree', 'kd_tree', 'brute'],    # mby use 'auto' ?
        # 'knc__leaf_size': [10, 128, 64],
        # 'knc__p': [1, 2, 5],
        # 'knc__metric': ['minkowski'],
        'knc__n_jobs': [1]
    }

    rfc_param_grid = {
        'pf__degree': [1, 2, 3],
        'pca__whiten': [True, False],
        'pca__n_components': n_components,
        # 'rfc__n_estimators': [10, 128, 64],
        # 'rfc__criterion': ['gini', 'entropy'],
        # 'rfc__max_features': ['auto', 'sqrt', 'log2', None],
        'rfc__n_jobs': [1]
    }

    # Best result 0.892
    etc_param_grid = {
        'pf__degree': [1, 2, 3],
        'pca__whiten': [True],  # [True, False],
        'pca__n_components': None,  # n_components,
        # 'etc__n_estimators': [1500],    # Memory consumption is fucking high... (~265 GB Ram needed)
        # 'etc__criterion': ['gini'],     # ['gini', 'entropy'],
        # 'etc__max_features': ['auto'],  # ['auto', 'sqrt', 'log2', None],
        'etc__n_jobs': [1]
    }

    gbc_param_grid = {
        'pf__degree': [1, 2, 3],
        'pca__whiten': [True, False],
        'pca__n_components': n_components,
        # 'gbc__loss': ['deviance', 'exponential'],
        # 'gbc__n_estimators': [10, 128, 100],
        # 'gbc__criterion': ['friedman_mse', 'mse', 'mae']
    }

    mlp_full_param_grid = {
        'pf__degree': [1, 2, 3],
        'pca__whiten': [True, False],
        'pca__n_components': n_components,
        # use a small layer in the middle (20-70% of the biggest) for countering overfitting ><><
        # 'mlp__hidden_layer_sizes': [(128,)],  # , (128, 64, 32, 16), (1024, 512, 256, 128)],
        # 'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
        # 'mlp__solver': ['lbfgs', 'sgd', 'adam'],
        # 'mlp__alpha': np.geomspace(1e-7, 1e2, 10),
        # 'mlp__learning_rate': ['constant', 'invscaling', 'adaptive'],
        # 'mlp__max_iter': [10000],
        # 'mlp__shuffle': [True, False],
        # 'mlp__random_state': [42],
        # 'mlp__tol': [1e-6]
    }

    estimators['mlp'] = mlp_estimator
    estimators['lsvc'] = lsvc_estimator
    estimators['svc'] = svc_estimator
    estimators['knc'] = knc_estimator
    estimators['rfc'] = rfc_estimator
    estimators['etc'] = etc_estimator
    estimators['gbc'] = gbc_estimator
    estimators['mlp_full'] = mlp_full_estimator

    param_grids['mlp'] = mlp_param_grid
    param_grids['lsvc'] = lsvc_param_grid
    param_grids['svc'] = svc_param_grid
    param_grids['knc'] = knc_param_grid
    param_grids['rfc'] = rfc_param_grid
    param_grids['etc'] = etc_param_grid
    param_grids['gbc'] = gbc_param_grid
    param_grids['mlp_full'] = mlp_full_param_grid

    # Scorer / Loss function
    acc = make_scorer(accuracy_score, greater_is_better=True)

    # GridsearchCV initializer
    grid_search_label = GridSearchCV(
        estimator=Pipeline(estimators[tech]),
        param_grid=param_grids[tech],
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

    # Save
    score_list.append(grid_search_label.best_score_)
    y_pred_list.append(grid_search_label.predict(X_train_unlabeled))

    print("======================================================================================")
    print("(iid, nfold): " + "(" + str(iid) + ", " + str(nfold) + ")")
    print("Best score:       " + str(grid_search_label.best_score_))
    print("Best parameters:   ")
    print()
    print(grid_search_label.best_params_)
    print()
    print("======================================================================================")


if __name__ == "__main__":

    # Get, split and transform train dataset
    data_train_labeled, train_labeled_index = read_hdf_to_matrix("train_labeled.h5")
    data_train_unlabeled, train_unlabeled_index = read_hdf_to_matrix("train_unlabeled.h5")
    data_test, test_index = read_hdf_to_matrix("test.h5")

    # Parameter search/evaluation
    old = []
    while data_train_labeled != old:
        old = data_train_labeled
        for tech in techs:
            y_pred_list = []
            score_list = []
            for iid in iids:
                for nfold in nfolds:
                    parameter_selection(data_train_labeled, data_train_unlabeled, data_test, nfold, iid, tech)

            # Get the prediction with the best score
            best_score = np.amax(score_list)
            best_score_index = score_list.index(best_score)
            y_pred_best_score.append(y_pred_list[best_score_index])

        for sample in range(np.size(y_pred_best_score, 1)):
            sample_list = []
            for result in range(np.size(y_pred_best_score, 0)):
                sample_list.append(y_pred_best_score[result][sample])
            ctr = col.Counter(sample_list)
            most_common = [(i, ctr[i] / len(sample_list) * 100.0) for i, count in ctr.most_common()][0]
            if most_common[1] >= 80:
                for result in range(np.size(y_pred_best_score, 0)):
                    if y_pred_best_score[result][sample] == most_common[0]:
                        index = y_pred_best_score[result].index(y_pred_best_score[result][sample])
                        data_train_labeled = np.r_[
                            data_train_labeled, np.c_[data_train_unlabeled[index][:], most_common[0]]]
                        np.delete(data_train_unlabeled, index, 0)
                        break

    y_pred_list = []
    score_list = []
    for tech in techs:
        for iid in iids:
            for nfold in nfolds:
                parameter_selection(data_train_labeled, data_train_unlabeled, data_test, nfold, iid, tech)

    # Get the prediction with the best score
    best_score = np.amax(score_list)
    best_score_index = score_list.index(best_score)
    y_pred_best_score.append(y_pred_list[best_score_index])

    # Print solution to file
    write_to_csv_from_vector("sample_franz_semi-supervised_too_simple.csv", test_index, y_pred_best_score)
