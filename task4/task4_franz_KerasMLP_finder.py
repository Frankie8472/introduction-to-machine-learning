# train.h5      - the training set
# test.h5       - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

# Library imports
import os
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "2"

import numpy as np
from pandas import read_hdf, DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Reshape, Dropout, Activation
from keras.optimizers import Nadam, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# Parameter initialization
estimators = {}
param_grids = {}
y_pred_list = []
score_list = []
features = 50


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


def baseline3_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(features,), activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    optimizer = Nadam(lr=0.002,
                      beta_1=0.9,
                      beta_2=0.999,
                      epsilon=1e-08,
                      schedule_decay=0.004)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    return model


def baseline2_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(features,), init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1024, init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, init='lecun_uniform'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model


def baseline4_model():
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


def baseline_model():
    # Create model
    model = Sequential()  # 'elu', 'relu', 'softmax', 'selu', 'softplus', 'softsign', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'
    model.add(Dense(features, input_dim=features, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_estimator():
    estimator = KerasClassifier(build_fn=baseline4_model, epochs=100, batch_size=100, verbose=0)
    return estimator


def evaluate(data_labeled):
    X, y = split_into_x_y(data_labeled)
    ss = StandardScaler()
    pca = PCA(n_components=features)
    transformed_X = pca.fit_transform(X)
    transformed_X = ss.fit_transform(transformed_X)
    skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=42)
    acc = make_scorer(accuracy_score, greater_is_better=True)
    estimator = get_estimator()
    results = cross_val_score(estimator, transformed_X, y, scoring=acc, cv=skf, n_jobs=2)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


def predict(data_labeled, X_test):
    X, y = split_into_x_y(data_labeled)
    pca = PCA(n_components=features)
    ss = StandardScaler()
    estimator = get_estimator()
    
    transformed_X = pca.fit_transform(X)
    transformed_X = ss.fit_transform(transformed_X)
    transformed_test = pca.transform(X_test)
    transformed_test = ss.transform(transformed_test)

    estimator.fit(transformed_X, y)

    y_pred = estimator.predict(transformed_test)

    # Print solution to file
    write_to_csv_from_vector("sample_franz_keras_mlp.csv", test_index, y_pred)


if __name__ == "__main__":
    # Get, split and transform train dataset
    data_train_labeled, train_labeled_index = read_hdf_to_matrix("train_labeled.h5")
    data_train_unlabeled, train_unlabeled_index = read_hdf_to_matrix("train_unlabeled.h5")
    data_test, test_index = read_hdf_to_matrix("test.h5")

    # Parameter search/evaluation
    # evaluate(data_train_labeled)
    predict(data_train_labeled, data_test)
