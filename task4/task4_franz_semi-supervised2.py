# train.h5      - the training set
# test.h5       - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import numpy as np
from pandas import read_hdf, DataFrame
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Parameter initialization
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


def go(Data_train_labeled, X_train_unlabeled, X_test):
    full_labeled, full_y = split_into_x_y(Data_train_labeled)
    full_unlabeled = X_train_unlabeled
    old_full_y_size = 0
    while old_full_y_size != np.size(full_y):
        print("hello")
        old_full_y_size = np.size(full_y)

        ss = StandardScaler()
        mlp = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation='relu',
            solver='lbfgs',
            alpha=13.0,
            learning_rate='constant',
            max_iter=200,
            shuffle=False,
            tol=1e-5
        )

        transformed_labeled = ss.fit_transform(full_labeled)
        transformed_unlabeled = ss.transform(full_unlabeled)

        mlp.fit(transformed_labeled, full_y)
        probability_unlabeled = mlp.predict_proba(transformed_unlabeled)
        predicted_unlabeled = mlp.predict(transformed_unlabeled)

        for i in range(np.size(probability_unlabeled, 0)):
            max_prob = np.amax(probability_unlabeled[i])
            max_prob_class = predicted_unlabeled[i]
            if max_prob >= 0.9:
                full_labeled = np.r_[full_labeled, [full_unlabeled[i]]]
                full_y = np.r_[full_y, max_prob_class]
                np.delete(full_unlabeled, i, 0)

    ss = StandardScaler()
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation='relu',
        solver='lbfgs',
        alpha=13.0,
        learning_rate='constant',
        max_iter=200,
        shuffle=False,
        tol=1e-5
    )

    transformed_labeled = ss.fit_transform(full_labeled)
    transformed_test = ss.transform(X_test)

    mlp.fit(transformed_labeled, full_y)
    y_pred_test = mlp.predict(transformed_test)

    # Print solution to file
    write_to_csv_from_vector("sample_franz_semi-supervised2.csv", test_index, y_pred_test)


if __name__ == "__main__":

    # Get, split and transform train dataset
    data_train_labeled, train_labeled_index = read_hdf_to_matrix("train_labeled.h5")
    data_train_unlabeled, train_unlabeled_index = read_hdf_to_matrix("train_unlabeled.h5")
    data_test, test_index = read_hdf_to_matrix("test.h5")

    # Parameter search/evaluation
    go(data_train_labeled, data_train_unlabeled, data_test)

