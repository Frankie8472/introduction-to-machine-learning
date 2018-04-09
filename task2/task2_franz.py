# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
from pandas import read_csv, DataFrame
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import GridSearchCV, ShuffleSplit


# Parameter initialization
random_state = 1
set = 4
n_jobs = 4
max_iter = 100000000

param_grid_logistic_reg = {
    #'tol': np.geomspace(0.001, 10, 10),
    'C': np.geomspace(0.001, 10, 10),
    'fit_intercept': [True, False],
    'solver': ['newton-cg', 'lbfgs',  'sag', 'saga'],   #'liblinear'
    'multi_class': ['ovr', 'multinomial']
}
param_grid_linear_SVC = {
    #'tol': np.geomspace(0.001, 1, 10),
    'C': np.geomspace(0.001, 10, 10),
    'multi_class': ['ovr', 'crammer_singer'],
    'fit_intercept': [True, False]
}

param_grid_SVC = {
    'C': np.geomspace(0.0000001, 0.0001, num=10),
    'kernel': ['poly'],   #['linear', 'poly', 'rbf', 'sigmoid'],  # 'precomputed'
    'degree': np.arange(3, 4),
    #'gamma': np.geomspace(0.000001, 1, num=100),
    #'tol': np.geomspace(0.1, 10, 5),
    'decision_function_shape': ['ova', 'ovr']
}

param_grid_NuSVC = {
    # 'nu': np.linspace(0.01, 1.0, 10),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # 'precomputed'
    'degree': np.arange(0, 5),
    # 'gamma': np.geomspace(0.001, 100, num=10),
    # 'tol': np.geomspace(0.001, 10, 5),
    'decision_function_shape': ['ovo', 'ovr']
}

param_grid_MLPClassifier = {
    'hidden_layer_sizes': (100,),
    'activation': ['tanh'],   #['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': np.geomspace(0.00001, 10, 5),
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': [1000],

}


# Define exercise functions
def read_csv_to_matrix(filename, index_name):
    data = read_csv(filename, index_col=index_name)
    return data.as_matrix(), data.index


def write_to_csv_from_vector(filename, vec):
    return DataFrame(vec).to_csv(filename, index=False, header=["Id", "y"])


def split_into_x_y(data_set):
    y = data_set[:, 0]
    x = data_set[:, 1:]
    return x, y


# Scorer / Loss function
acc = make_scorer(accuracy_score, greater_is_better=True)

# Estimator / Predictor
clfs = [
    # ('KNeighborsClf', KNeighborsClassifier()),
    # ('DecisionTreeClf', DecisionTreeClassifier()),
    # ('MLPClf', MLPClassifier(hidden_layer_sizes=100000, activation="relu", learning_rate="adaptive")),
    # ('RandomForestClf', RandomForestClassifier()),
    # ('ExtraTreesClf', ExtraTreesClassifier()),
    # ('BaggingClf', BaggingClassifier()),
    ('LogisticReg', param_grid_logistic_reg, LogisticRegression(
        tol=0.0001,
        C=1.0,
        fit_intercept=True,
        random_state=random_state,
        solver='newton-cg',
        max_iter=max_iter,
        multi_class='ovr',
        verbose=0,
        warm_start=False,
        n_jobs=n_jobs)),

    ('LinearSVC', param_grid_linear_SVC, LinearSVC(
        dual=False,
        tol=0.0001,
        C=1.0,
        multi_class='ovr',
        fit_intercept=True,
        verbose=0,
        random_state=random_state,
        max_iter=max_iter)),

    ('SVC', param_grid_SVC, SVC(
        C=1.0,
        kernel='rbf',
        degree=3,
        gamma='auto',
        coef0=0.0,
        probability=False,
        shrinking=True,
        tol=0.001,
        decision_function_shape='ovr',
        random_state=random_state)),

    ('NuSVC', param_grid_NuSVC, NuSVC(
        nu=0.5,
        kernel='rbf',
        degree=3,
        gamma='auto',
        coef0=0.0,
        probability=False,
        shrinking=True,
        tol=0.001,
        verbose=False,
        max_iter=max_iter,
        decision_function_shape='ovr',
        random_state=random_state)),

    ('MLPClassifier', param_grid_MLPClassifier, MLPClassifier())
]

# Get, split and transform train dataset
data_train, index_train = read_csv_to_matrix("train.csv", "Id")
X_train, y_train = split_into_x_y(data_train)

# Get, split and transform test dataset
X_test, index_test = read_csv_to_matrix("test.csv", "Id")

for name, param_grid, clf in [clfs[set]]:
    # Grid
    clf_searcher = GridSearchCV(estimator=clf,
                                param_grid=param_grid,
                                scoring=acc,
                                n_jobs=n_jobs,
                                iid=False,
                                cv=3,
                                refit=True,
                                verbose=10
                                )

    # Train
    clf_searcher.fit(X_train, y_train)

    # Predict
    y_pred = np.int8(clf_searcher.predict(X_test))

    print(clf_searcher.best_score_)

    # Print solution to file
    write_to_csv_from_vector("sample_franz_" + name + ".csv", np.c_[index_test, y_pred])
