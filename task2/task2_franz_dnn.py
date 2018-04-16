# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
from pandas import read_csv, DataFrame
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.decomposition import PCA

# Parameter initialization
cores = 48  # Number of cores for parallelization
message_count = 10  # Bigger = More msgs


# Define exercise functions
def read_csv_to_matrix(filename, index_name):
    data = read_csv("machine_learning/input/" + filename, index_col=index_name)
    return data.as_matrix(), data.index


def write_to_csv_from_vector(filename, vec):
    return DataFrame(vec).to_csv("output/" + filename, index=False, header=["Id", "y"])


def split_into_x_y(data_set):
    y = data_set[:, 0]
    x = data_set[:, 1:]
    return x, y


def preprocessing(X, y):
    data_sets = []

    scalerSTD = StandardScaler(
        copy=True,
        with_mean=True,
        with_std=True
    )

    scalerPCA1 = PCA(
        # n_components=,
        copy=True,
        whiten=False,
        svd_solver='auto',
        # tol=.0,
        iterated_power='auto',
    )

    scalerPCA2 = PCA(
        # n_components=,
        copy=True,
        whiten=True,
        svd_solver='auto',
        # tol=.0,
        iterated_power='auto',
    )

    scalerPoly1 = PolynomialFeatures(
        degree=3,
        interaction_only=False,
        include_bias=True
    )

    scalerPoly2 = PolynomialFeatures(
        degree=3,
        interaction_only=False,
        include_bias=False
    )

    scalerMinMax = MinMaxScaler(
        feature_range=(0, 1),
        copy=True
    )

    data_sets.append(('Normal', X))
    data_sets.append(('STD', scalerSTD.fit_transform(X)))
    data_sets.append(('PCA1', scalerPCA1.fit_transform(X)))
    data_sets.append(('PCA2', scalerPCA2.fit_transform(X)))
    data_sets.append(('Poly1', scalerPoly1.fit_transform(X)))
    data_sets.append(('Poly2', scalerPoly2.fit_transform(X)))
    data_sets.append(('MinMax', scalerMinMax.fit_transform(X)))

    return data_sets


def mlp_param_selection(set_name, X, y, nfold, iid):
    param_grid = {
        'hidden_layer_sizes': [(10,), (20,), (16, 8, 4, 2), (14, 12, 10, 8, 6, 4, 2)],
        # 'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        # 'solver': ['lbfgs', 'sgd', 'adam']
        'solver': ['lbfgs', 'sgd'],     # ['lbfgs', 'sgd', 'adam']
        'alpha': np.linspace(1, 20, 10),
        # 'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [100000],
        # 'shuffle': [True, False],
        'tol': np.geomspace(10 ** -20, 10 ** -10, 11)
    }

    # Scorer / Loss function
    acc = make_scorer(accuracy_score, greater_is_better=True)

    # Estimator / Predictor
    clf = MLPClassifier(
        hidden_layer_sizes=(100, 100, 100, 100),  # 4 Hidden layers with 100 perceptrons each
        activation='tanh',
        solver='lbfgs',
        alpha=0.0001,
        # batch_size='auto',
        learning_rate='constant',
        # learning_rate_init=0.001,
        # power_t=0.5,
        max_iter=1000,
        shuffle=False,
        random_state=8,
        tol=10 ** -4,
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
        verbose=message_count,
        error_score='raise',
        return_train_score=False,
    )

    # Train
    grid_search.fit(X, y)

    print("\n======================================================================================")
    print("For (set_name, iid, nfold): " + "(" + set_name + ", " + str(iid) + ", " + str(nfold) + ")")
    print("Best score:       " + str(grid_search.best_score_))
    print("Best score refit: " + str(grid_search.score(X, y)))
    print("Best parameter:   ")
    print("")
    print(grid_search.best_params_)
    print("")
    print("======================================================================================\n")


# Get, split and transform train dataset
data_train, index_train = read_csv_to_matrix("train.csv", "Id")
X_train, y_train = split_into_x_y(data_train)

# Preprocessing
data_sets = preprocessing(X_train, y_train)

# Find best parameters

for set_name, data_set in data_sets:
    for iid in [True, False]:
        for nfold in [5, 10]:
            mlp_param_selection(set_name, data_set, y_train, nfold, iid)

# Get, split and transform test dataset
# X_test, index_test = read_csv_to_matrix("test.csv", "Id")

# Predict
# y_pred = np.int8(clf_searcher.predict(X_test))

# Print solution to file
# write_to_csv_from_vector("sample_franz_dnn.csv", np.c_[index_test, y_pred])
