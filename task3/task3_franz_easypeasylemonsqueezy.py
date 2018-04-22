# train.h5      - the training set
# test.h5       - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

# !-----------------------------!
#  RUN ONLY ON EULER WITH run.sh
# !-----------------------------!

import numpy as np
from pandas import read_hdf, DataFrame
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":

    # Parameter initialization
    cores = 2               # Number of cores for parallelization
    message_count = 2       # Bigger = More msgs
    nfolds = [10]
    iids = [False]
    estimators = {}
    param_grids = {}
    y_pred_list = []
    score_list = []
    data_sets = {}

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


    def parameter_selection(data_train, X_test, nfold, iid):

        # Initialize feature tranformer
        ss = StandardScaler()
        clf = MLPClassifier()

        mlp_param_grid = {
            'hidden_layer_sizes': [(100,)],
            'activation': ['tanh'],
            'solver': ['lbfgs'],
            'alpha': np.linspace(23, 25, 3),
            'learning_rate': ['constant'],
            'max_iter': [100],
            'shuffle': [False],
            'random_state': [42],
            'tol': [1e-4]
        }

        # Scorer / Loss function
        acc = make_scorer(accuracy_score, greater_is_better=True)

        # GridsearchCV initializer
        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=mlp_param_grid,
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

        # Prepare data
        X_train, y_train = split_into_x_y(data_train)

        # Scale data
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)

        # Train
        grid_search.fit(X_train, y_train)

        # Predict
        y_pred = grid_search.predict(X_test)

        # Save
        score_list.append(grid_search.best_score_)
        y_pred_list.append(y_pred)

        print("======================================================================================")
        print("(iid, nfold): " + "(" + str(iid) + ", " + str(nfold) + ")")
        print("Best score:       " + str(grid_search.best_score_))
        print("Best parameters:   ")
        print("")
        print(grid_search.best_params_)
        print("")
        print("======================================================================================")


    # Get, split and transform train dataset
    data_train, train_index = read_hdf_to_matrix("train.h5")
    data_test, test_index = read_hdf_to_matrix("test.h5")

    # Parameter search/evaluation
    for iid in iids:
        for nfold in nfolds:     # for leave-one-out: np.size(X_train, 0)
            parameter_selection(data_train, data_test, nfold, iid)

    # Get the prediction with the best score
    best_score = np.amax(score_list)
    best_score_index = score_list.index(best_score)
    y_pred_best_score = y_pred_list[best_score_index]

    # Print solution to file
    write_to_csv_from_vector("sample_franz.csv", test_index, y_pred_best_score)
