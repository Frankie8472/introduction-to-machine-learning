# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
from pandas import read_csv, DataFrame
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

# Parameter initialization
cores = 4
sol = []
scores = []


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


def mlp_param_selection(X_test, X_train, y_train, nfold, iid):
    param_grid = {
        # 'hidden_layer_sizes': [(10,), (20,), (16, 8, 4, 2), (14, 12, 10, 8, 6, 4, 2)],
        # 'activation': ['identity', 'logistic', 'tanh', 'relu'],
        # 'activation': ['identity', 'logistic', 'tanh', 'relu'],
        # 'solver': ['lbfgs', 'sgd', 'adam']
        # 'solver': ['lbfgs', 'sgd'],     # ['lbfgs', 'sgd', 'adam']
        # 'alpha': np.linspace(1, 20, 10),
        # 'learning_rate': ['constant', 'invscaling', 'adaptive'],
        # 'max_iter': [100000],
        # 'shuffle': [True, False],
        # 'tol': np.geomspace(10 ** -20, 10 ** -10, 11)
    }

    # Scorer / Loss function
    acc = make_scorer(accuracy_score, greater_is_better=True)

    # Estimator / Predictor
    clf = MLPClassifier(
        hidden_layer_sizes=(20,),  # 4 Hidden layers with 100 perceptrons each
        activation='tanh',
        solver='lbfgs',
        alpha=16.0,
        # batch_size='auto',
        learning_rate='constant',
        # learning_rate_init=0.001,
        # power_t=0.5,
        max_iter=100000,
        shuffle=False,
        random_state=None,
        tol=10 ** -10,
        # verbose=False,
        # warm_start=False,
        # momentum=0.9,
        # nesterovs_momentum=True,
        # early_stopping=False,
        # validation_fraction=0.1,
        # beta_1=0.9,
        # beta_2=0.999,
        # epsilon=10**-8
    )

    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring=acc,
        n_jobs=cores,
        pre_dispatch='2*n_jobs',
        iid=iid,
        cv=nfold,
        refit=True,
        verbose=0,
        error_score='raise',
        return_train_score=False
    )

    # Train
    grid_search.fit(X_train, y_train)
    scores.append(grid_search.best_score_)
    y_pred = grid_search.predict(X_test)
    sol.append(y_pred)
    grid_search.refit


# Get, split and transform train dataset
data, index_train = read_csv_to_matrix("train.csv", "Id")
X_train, y_train = split_into_x_y(data)
data_test, index_test = read_csv_to_matrix("test.csv", "Id")

for cv in [10]:
    mlp_param_selection(data_test, X_train, y_train, cv, True)


# Get best
print(np.amax(scores))
y_pred = sol[scores.index(np.amax(scores))]

# Print solution to file
write_to_csv_from_vector("sample_franz_dnn_final_cv.csv", np.c_[index_test, y_pred])
