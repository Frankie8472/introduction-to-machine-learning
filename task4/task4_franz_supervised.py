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
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":

    # Parameter initialization
    cores = 3              # Number of cores for parallelization
    message_count = 2      # Bigger = More msgs
    techs = ['mlp'],       # 'mlp', 'lsvc', 'svc', 'knc', 'rfc', 'etc', 'gbc'
    whiten = [True, False]
    n_components = [None, 0.95, 0.97, 0.99]
    nfolds = [3, 5, 10]     # try out 5 and 10
    iids = [True, False]
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


    def parameter_selection(data_train_labeled, X_test, nfold, iid, tech):
        X_train_labeled, y_train_labeled = split_into_x_y(data_train_labeled)

        mlp_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('mlp', MLPClassifier())
        ]

        lsvc_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('lsvc', LinearSVC())
        ]

        svc_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('svc', SVC())
        ]

        knc_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('knc', KNeighborsClassifier())
        ]

        rfc_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('rfc', RandomForestClassifier())
        ]

        etc_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('etc', ExtraTreesClassifier())
        ]

        gbc_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('etc', GradientBoostingClassifier())
        ]

        mlp_param_grid = {
            'pca__whiten': [True, False],
            'pca__n_components': [None, 0.95, 0.97, 0.99],
        }

        lsvc_param_grid = {
            'pca__whiten': [True, False],
            'pca__n_components': [None, 0.95, 0.97, 0.99],
        }

        svc_param_grid = {
            'pca__whiten': [True, False],
            'pca__n_components': [None, 0.95, 0.97, 0.99],
        }

        knc_param_grid = {
            'pca__whiten': [True, False],
            'pca__n_components': [None, 0.95, 0.97, 0.99],
        }

        rfc_param_grid = {
            'pca__whiten': [True, False],
            'pca__n_components': [None, 0.95, 0.97, 0.99],
        }

        etc_param_grid = {
            'pca__whiten': [True, False],
            'pca__n_components': [None, 0.95, 0.97, 0.99],
        }

        gbc_param_grid = {
            'pca__whiten': [True, False],
            'pca__n_components': [None, 0.95, 0.97, 0.99],
        }

        # 'mlp', 'lsvc', 'svc', 'knc', 'rfc', 'etc', 'gbc'

        estimators['mlp'] = mlp_estimator
        estimators['lsvc'] = lsvc_estimator
        estimators['svc'] = svc_estimator
        estimators['knc'] = knc_estimator
        estimators['rfc'] = rfc_estimator
        estimators['etc'] = etc_estimator
        estimators['gbc'] = gbc_estimator

        param_grids['mlp'] = mlp_param_grid
        param_grids['lsvc'] = lsvc_param_grid
        param_grids['svc'] = svc_param_grid
        param_grids['knc'] = knc_param_grid
        param_grids['rfc'] = rfc_param_grid
        param_grids['etc'] = etc_param_grid
        param_grids['gbc'] = gbc_param_grid

        # Scorer / Loss function
        acc = make_scorer(accuracy_score, greater_is_better=True)

        # GridsearchCV initializer
        grid_search = GridSearchCV(
            estimator=Pipeline(estimators[tech]),
            param_grid=param_grids[tech],
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
        grid_search.fit(X_train_labeled, y_train_labeled)

        # Save
        score_list.append(grid_search.best_score_)
        y_pred_list.append(grid_search.predict(X_test))

        print("======================================================================================")
        print("(iid, nfold, tech): " + "(" + str(iid) + ", " + str(nfold) + ", " + str(tech) + ")")
        print("Best score:       " + str(grid_search.best_score_))
        print("Best parameters:   ")
        print()
        print(grid_search.best_params_)
        print()
        print("======================================================================================")


    # Get, split and transform train dataset
    data_train_labeled, train_labeled_index = read_hdf_to_matrix("train_labeled.h5")
    data_test, test_index = read_hdf_to_matrix("test.h5")

    # Parameter search/evaluation
    for tech in techs:
        for iid in iids:
            for nfold in nfolds:
                parameter_selection(data_train_labeled, data_test, nfold, iid, tech)

    # Get the prediction with the best score
    best_score = np.amax(score_list)
    best_score_index = score_list.index(best_score)
    y_pred_best_score = y_pred_list[best_score_index]

    # Print solution to file
    write_to_csv_from_vector("sample_franz_supervised.csv", test_index, y_pred_best_score)
