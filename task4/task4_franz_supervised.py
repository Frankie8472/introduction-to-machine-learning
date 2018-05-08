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
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":

    # Parameter initialization
    cores = 3              # Number of cores for parallelization
    message_count = 2       # Bigger = More msgs
    techs = ['lp', 'ls']    # 'lp', 'ls'
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


    def parameter_selection(X_train_labeled, y_train_labeled, X_test, nfold, iid, n_components, whiten, tech):
        ss = StandardScaler()
        pca = PCA(n_components=n_components,
                  whiten=whiten
                  )

        X_train_labeled = ss.fit_transform(X_train_labeled)
        X_test = ss.transform(X_test)

        X_train_labeled = pca.fit_transform(X_train_labeled)
        X_test = pca.transform(X_test)

        lp_estimator = [
            ('mlp', MLPClassifier())
        ]

        ls_estimator = [
            ('ls', LabelSpreading())
        ]

        mlp_param_grid = {
            'lp__kernel': ['knn'], #, 'rbf'],   # or callable kernel function
            'gamma': np.geomspace(1e-3, 1e3, num=7),
            'lp__n_neighbors': [1, 5, 10, 50, 100],
            'lp__max_iter': [10000],
            'lp__tol': [1e-6],
            'lp__n_jobs': [cores]
        }

        ls_param_grid = {
            'ls__kernel': ['knn'],#, 'rbf'],   # or callable kernel function
            'ls__gamma': np.geomspace(1e-3, 1e3, num=7),
            'n_neighbors': [1, 5, 10, 50, 100],
            'ls__max_iter': [10000],
            'ls__tol': [1e-6],
            'ls__n_jobs': [cores]
        }

        estimators['mlp'] = mlp_estimator
        estimators['ls'] = ls_estimator

        param_grids['mlp'] = mlp_param_grid
        param_grids['ls'] = ls_param_grid

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
        grid_search.fit(X_train_labeled, full_y_set)

        # Predict
        y_pred = grid_search.predict(X_test)

        # Save
        score_list.append(grid_search.best_score_)
        y_pred_list.append(y_pred)

        print("======================================================================================")
        print("(iid, nfold, n_components, whiten, tech): " + "(" + str(iid) + ", " + str(nfold) + ", " + str(n_components) + ", " + str(whiten) + ", " + str(tech) + ")")
        print("Best score:       " + str(grid_search.best_score_))
        print("Best parameters:   ")
        print("")
        print(grid_search.best_params_)
        print("")
        print("======================================================================================")


    # Get, split and transform train dataset
    data_train_labeled, train_labeled_index = read_hdf_to_matrix("train_labeled.h5")
    data_test, test_index = read_hdf_to_matrix("test.h5")

    X_train_labeled, y_train_labeled = split_into_x_y(data_train_labeled)

    # Parameter search/evaluation
    for tech in techs:
        for component in n_components:
            for white in whiten:
                for iid in iids:
                    for nfold in nfolds:     # for leave-one-out: np.size(X_train, 0)
                        parameter_selection(X_train_labeled, y_train_labeled, data_test, nfold, iid, component, white, tech)

    # Get the prediction with the best score
    best_score = np.amax(score_list)
    best_score_index = score_list.index(best_score)
    y_pred_best_score = y_pred_list[best_score_index]

    # Print solution to file
    write_to_csv_from_vector("sample_franz_supervised.csv", test_index, y_pred_best_score)
