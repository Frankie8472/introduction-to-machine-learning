# train_labeled.h5      - the labeled training set
# train_unlabeled.h5    - the unlabeled training set
# test.h5               - the test set (make predictions based on this file)
# sample.csv            - a sample submission file in the correct format

# !-----------------------------!
#  RUN ONLY ON EULER WITH run.sh
# !-----------------------------!

import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import read_hdf, DataFrame
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    BaggingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold

features = 60


def get_estimator():
    global features
    features = 60

    return RandomForestClassifier(n_estimators=1000, n_jobs=2, min_samples_leaf=2, min_samples_split=3,criterion="entropy")
    return BaggingClassifier(KNeighborsClassifier(n_neighbors=1, leaf_size=5, p=2, n_jobs=2), max_samples=0.8,
                             max_features=0.8)  # 128 -> 0.82
    return LinearSVC(C=4.0, fit_intercept=True, tol=1e-4)  # 128 -> 0.8605
    return KNeighborsClassifier(n_neighbors=1, leaf_size=5, p=2, n_jobs=2)  # 110 -> 0.8229
    return ExtraTreesClassifier(n_estimators=1000)
    return KNeighborsClassifier(n_neighbors=1)
    return BaggingClassifier(SVC(), max_samples=0.3, max_features=0.9)
    return PassiveAggressiveClassifier(max_iter=4000, tol=1e-5, C=1e-5, shuffle=False)  # 0.85
    return SGDClassifier(max_iter=500, shuffle=False)  # 0.85
    return LogisticRegression()  # 128, 86%
    return SVC(C=3.0, kernel='poly', degree=3)
    return NuSVC(nu=0.1)  # 56 features -> 0.92
    return OneVsRestClassifier(LinearSVC(C=10.0))  # 0.84
    return GradientBoostingClassifier(n_estimators=1000)
    return SVC(C=3.0)  # 65 features -> 0.92
    return KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=100, verbose=0)
    return MLPClassifier(
        hidden_layer_sizes=(400, 400),
        activation='relu',
        solver='lbfgs',
        alpha=13.0,
        learning_rate='constant',
        max_iter=200,
        shuffle=False,
        tol=1e-4
    )  # 60 -> 92.37

    return VotingClassifier(voting='hard',
                            estimators=[
                                ('svc', SVC(C=3.0, probability=True)),
                                ('nusvc', NuSVC(nu=0.1, probability=True)),
                                ('keras',
                                 KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=100, verbose=0)),
                                ('mlp', MLPClassifier(
                                    hidden_layer_sizes=(400, 400, 400),
                                    activation='relu',
                                    solver='lbfgs',
                                    alpha=13.0,
                                    learning_rate='constant',
                                    max_iter=300,
                                    shuffle=False,
                                    tol=1e-5
                                ))
                            ]
                            )


def baseline_model():
    # Create model
    model = Sequential()  # 'elu', 'relu', 'softmax', 'selu', 'softplus', 'softsign', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'
    model.add(Dense(400, input_dim=features, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


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


def evaluate(data_labeled):
    X, y = split_into_x_y(data_labeled)

    estimator = get_estimator()
    ss = StandardScaler()
    pca = PCA(n_components=features)

    transformed_X = pca.fit_transform(X)
    transformed_X = ss.fit_transform(transformed_X)

    skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=42)
    acc = make_scorer(accuracy_score, greater_is_better=True)
    results = cross_val_score(estimator, transformed_X, y, scoring=acc, cv=skf, n_jobs=5)

    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


def predict(data_labeled, X_unlabeled, X_test):
    X_labeled, y_labeled = split_into_x_y(data_labeled)

    mlp = get_estimator()
    ss = StandardScaler()
    pca = PCA(n_components=features)

    transformed_X_labeled = pca.fit_transform(X_labeled)
    transformed_X_labeled = ss.fit_transform(transformed_X_labeled)
    transformed_X_unlabeled = pca.transform(X_unlabeled)
    transformed_X_unlabeled = ss.transform(transformed_X_unlabeled)

    mlp.fit(transformed_X_labeled, y_labeled)
    y_unlabeled = mlp.predict(transformed_X_unlabeled)

    transformed_X = np.r_[X_labeled, X_unlabeled]
    transformed_y = np.r_[y_labeled, y_unlabeled]

    mlp = get_estimator()
    ss = StandardScaler()
    pca = PCA(n_components=features)

    transformed_X = pca.fit_transform(transformed_X)
    transformed_X = ss.fit_transform(transformed_X)
    transformed_test = pca.transform(X_test)
    transformed_test = ss.transform(transformed_test)

    mlp.fit(transformed_X, transformed_y)
    y_pred = mlp.predict(transformed_test)

    # Print solution to file
    write_to_csv_from_vector("sample_franz_ensemble_voting_hard.csv", data_test_index, y_pred)


if __name__ == "__main__":
    # Get, split and transform train dataset
    data_train_labeled, data_train_labeled_index = read_hdf_to_matrix("train_labeled.h5")
    data_train_unlabeled, train_unlabeled_index = read_hdf_to_matrix("train_unlabeled.h5")
    data_test, data_test_index = read_hdf_to_matrix("test.h5")

    evaluate(data_train_labeled)
    # predict(data_train_labeled, data_train_unlabeled, data_test)
