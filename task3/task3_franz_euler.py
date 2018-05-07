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
from sklearn.kernel_approximation import RBFSampler, Nystroem
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RepeatedEditedNearestNeighbours, ClusterCentroids
from imblearn.combine import SMOTEENN, SMOTETomek

if __name__ == "__main__":

    # Parameter initialization
    cores = 48              # Number of cores for parallelization
    message_count = 1       # Bigger = More msgs
    tech = 'mlp'            # 'mlp', 'sgd', 'rbf' or 'ny'
    nfolds = [3, 5, 10]
    iids = [True, False]
    estimators = {}
    param_grids = {}
    y_pred_list = []
    score_list = []
    data_set = []

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


    def parameter_selection(set_name, X_train, y_train, X_test, nfold, iid):
        mlp_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('mlp', MLPClassifier())
        ]

        sgd_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('sgd', SGDClassifier())
        ]

        ny_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('ny', Nystroem()),
            ('sgd', SGDClassifier())
        ]

        rbf_estimator = [
            ('ss', StandardScaler()),
            ('pca', PCA()),
            ('rbf', RBFSampler()),
            ('sgd', SGDClassifier())
        ]

        mlp_param_grid = {
            'pca__n_components': [None, 0.95],
            'pca__whiten': [True, False],
            'mlp__hidden_layer_sizes': [(100,), (50,), (100, 50, 25), (25, 50, 100, 50, 25)],
            'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'mlp__solver': ['lbfgs', 'sgd', 'adam'],
            'mlp__alpha': np.geomspace(1e-7, 1e2, 10),
            'mlp__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'mlp__max_iter': [100000],
            'mlp__shuffle': [True, False],
            'mlp__random_state': [42],
            'mlp__tol': [1e-6]
        }

        sgd_param_grid = {
            'pca__n_components': [None, 0.95],
            'pca__whiten': [True, False],
            'sgd__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon'],
            'sgd__penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'sgd__alpha': np.geomspace(1e-7, 1e2, 10),
            'sgd__l1_ratio': np.linspace(0, 1, num=11),
            'sgd__fit_intercept': [True, False],
            'sgd__max_iter': [10000],
            'sgd__tol': [1e-6],
            'sgd__shuffle': [True, False],
            'sgd__njobs': [cores],
            'sgd__random_state': [42],
            'sgd__learning_rate': ['constant', 'optimal', 'invscaling']
        }

        ny_param_grid = {
            'pca__n_components': [None, 0.95],
            'pca__whiten': [True, False],
            'ny__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'ny__n_components': [100],
            'ny__degree': [1, 2, 3],
            'sgd__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon'],
            'sgd__penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'sgd__alpha': np.geomspace(1e-7, 1e2, 10),
            'sgd__l1_ratio': np.linspace(0, 1, num=11),
            'sgd__fit_intercept': [True, False],
            'sgd__max_iter': [10000],
            'sgd__tol': [1e-6],
            'sgd__shuffle': [True, False],
            'sgd__njobs': [cores],
            'sgd__random_state': [42],
            'sgd__learning_rate': ['constant', 'optimal', 'invscaling']
        }

        rbf_param_grid = {
            'pca__n_components': [None, 0.95],
            'pca__whiten': [True, False],
            'rbf__gamma': np.geomspace(1e-1, 1e1, 3),
            'rbf__n_components': [100],
            'sgd__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon'],
            'sgd__penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'sgd__alpha': np.geomspace(1e-7, 1e2, 10),
            'sgd__l1_ratio': np.linspace(0, 1, num=11),
            'sgd__fit_intercept': [True, False],
            'sgd__max_iter': [10000],
            'sgd__tol': [1e-6],
            'sgd__shuffle': [True, False],
            'sgd__njobs': [cores],
            'sgd__random_state': [42],
            'sgd__learning_rate': ['constant', 'optimal', 'invscaling']
        }
        estimators['mlp'] = mlp_estimator
        estimators['sgd'] = sgd_estimator
        estimators['rbf'] = rbf_estimator
        estimators['ny'] = ny_estimator

        param_grids['mlp'] = mlp_param_grid
        param_grids['sgd'] = sgd_param_grid
        param_grids['rbf'] = rbf_param_grid
        param_grids['ny'] = ny_param_grid

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
        grid_search.fit(X_train, y_train)

        # Predict
        y_pred = grid_search.predict(X_test)

        # Save
        score_list.append(grid_search.best_score_)
        y_pred_list.append(y_pred)

        print("======================================================================================")
        print("(set_name, iid, nfold): " + "(" + str(set_name) + ", " + str(iid) + ", " + str(nfold) + ")")
        print("Best score:       " + str(grid_search.best_score_))
        print("Best parameters:   ")
        print("")
        print(grid_search.best_params_)
        print("")
        print("======================================================================================")


    # Get, split and transform train dataset
    data_train, train_index = read_hdf_to_matrix("train.h5")
    data_test, test_index = read_hdf_to_matrix("test.h5")

    X, y = split_into_x_y(data_train)
    adasyn = ADASYN(ratio='minority', n_jobs=cores)
    smote = SMOTE(n_jobs=cores)
    ros = RandomOverSampler()
    nearmiss = NearMiss(n_jobs=cores)
    renn = RepeatedEditedNearestNeighbours(n_jobs=cores)
    clustercentroids = ClusterCentroids(n_jobs=cores)
    smoteenn = SMOTEENN(n_jobs=cores)
    smotetomek = SMOTETomek(n_jobs=cores)

    X_adasyn, y_adasyn = adasyn.fit_sample(X, y)
    X_smote, y_smote = smote.fit_sample(X, y)
    X_ros, y_ros = RandomOverSampler = ros.fit_sample(X, y)
    X_nearmiss, y_nearmiss = nearmiss.fit_sample(X, y)
    X_renn, y_renn = renn.fit_sample(X, y)
    X_clustercentroids, y_clustercentroids = clustercentroids.fit_sample(X, y)
    X_smoteenn, y_smoteenn = smoteenn.fit_sample(X, y)
    X_smotetomek, y_smotetomek = smotetomek.fit_sample(X, y)

    data_set.append(("normal", X, y))
    data_set.append(("adasyn", X_adasyn, y_adasyn))
    data_set.append(("smote", X_smote, y_smote))
    data_set.append(("ros", X_ros, y_ros))
    data_set.append(("nearmiss", X_nearmiss, y_nearmiss))
    data_set.append(("renn", X_renn, y_renn))
    data_set.append(("clustercentroids", X_clustercentroids, y_clustercentroids))
    data_set.append(("smoteenn", X_smoteenn, y_smoteenn))
    data_set.append(("smotetomek", X_smotetomek, y_smotetomek))

    print(np.size(y))
    print(np.size(y_smote))
    print(np.size(y_ros))
    print(np.size(y_nearmiss))
    print(np.size(y_clustercentroids))
    print(np.size(y_smoteenn))
    print(np.size(y_smotetomek))

    # Parameter search/evaluation
    for set_name, X_train, y_train in data_set:
        for iid in iids:
            for nfold in nfolds:     # for leave-one-out: np.size(X_train, 0)
                parameter_selection(set_name, X_train, y_train, data_test, nfold, iid)

    # Get the prediction with the best score
    best_score = np.amax(score_list)
    best_score_index = score_list.index(best_score)
    y_pred_best_score = y_pred_list[best_score_index]

    # Print solution to file
    write_to_csv_from_vector("sample_franz.csv", test_index, y_pred_best_score)
