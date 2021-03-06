# train.h5      - the training set
# test.h5       - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

# Library imports
import os

from sklearn.decomposition import PCA

os.environ["HDF5_DISABLE_VERSION_CHECK"] = "2"

import numpy as np
from pandas import read_hdf, DataFrame
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

# Parameter initialization
estimators = {}
param_grids = {}
y_pred_list = []
score_list = []
features = 50
max_iter = 3


# Define exercise functions
def read_hdf_to_matrix(filename):
    data = read_hdf("input/" + filename)
    return data.as_matrix(), data.index


def write_to_csv_from_vector(filename, index_col, vec):
    return DataFrame(np.c_[index_col, vec]).to_csv("output/" + filename, index=False, header=["Id", "y"])


def split_into_x_y(data_set):
    y = data_set[:, 0]
    X = data_set[:, 1:]
    return X, y


def baseline_model():
    # Create model
    model = Sequential()
    model.add(Dense(400, input_dim=features, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_estimator():
    estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=100, verbose=0)
    return estimator


def go(Data_train_labeled, X_train_unlabeled, X_test):
    full_labeled, full_y = split_into_x_y(Data_train_labeled)
    full_unlabeled = X_train_unlabeled
    old_full_y_size = 0
    count = 0
    while (old_full_y_size != np.size(full_y)) & (count < max_iter):
        count = count + 1
        print(old_full_y_size)
        old_full_y_size = np.size(full_y)

        pca = PCA(n_components=features)
        ss = StandardScaler()
        mlp = get_estimator()

        transformed_labeled = pca.fit_transform(full_labeled)
        transformed_labeled = ss.fit_transform(transformed_labeled)
        transformed_unlabeled = pca.transform(full_unlabeled)
        transformed_unlabeled = ss.transform(transformed_unlabeled)

        mlp.fit(transformed_labeled, full_y)
        probability_unlabeled = mlp.predict_proba(transformed_unlabeled)
        predicted_unlabeled = mlp.predict(transformed_unlabeled)

        for i in range(np.size(probability_unlabeled, 0)):
            max_prob = np.amax(probability_unlabeled[i])
            max_prob_class = predicted_unlabeled[i]
            if max_prob >= 0.99:
                full_labeled = np.r_[full_labeled, [full_unlabeled[i]]]
                full_y = np.r_[full_y, max_prob_class]
                np.delete(full_unlabeled, i, 0)

    pca = PCA(n_components=features)
    ss = StandardScaler()
    mlp = get_estimator()

    transformed_labeled = pca.fit_transform(full_labeled)
    transformed_labeled = ss.fit_transform(transformed_labeled)
    transformed_test = pca.transform(X_test)
    transformed_test = ss.transform(transformed_test)

    mlp.fit(transformed_labeled, full_y)
    y_pred_test = mlp.predict(transformed_test)

    # Print solution to file
    write_to_csv_from_vector("sample_franz_semi-supervised_with_Keras.csv", test_index, y_pred_test)


if __name__ == "__main__":
    # Get, split and transform train dataset
    data_train_labeled, train_labeled_index = read_hdf_to_matrix("train_labeled.h5")
    data_train_unlabeled, train_unlabeled_index = read_hdf_to_matrix("train_unlabeled.h5")
    data_test, test_index = read_hdf_to_matrix("test.h5")

    # Parameter search/evaluation
    go(data_train_labeled, data_train_unlabeled, data_test)
