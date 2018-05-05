
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from pandas import read_hdf
from pandas import read_csv

def get_train_data(filename):
    train_data = read_hdf("input/"+filename)
    train_data = train_data.as_matrix() #note: .as_matrix() remoxes labeling columns
    x_train = train_data[0:, 1:] # coll 1 to inf
    y_train = train_data[0:, 0] # first col
    return y_train, x_train

def get_test_data(filename):
    test_data = read_hdf("input/"+filename)
    test_data = test_data.as_matrix() #note: .as_matrix() remoxes labeling columns
    x_test = test_data[0:, 0:] # coll 0 to inf
    return x_test

def get_index():
    test_data = read_hdf("input/test.h5")
    index = test_data.index
    print(index,'\n')
    print(type(index))
    return index

def write_test_to_csv(y):
    i = get_index()
    print(type(y))

    print(type(i))

    sol = np.c_[i, y]
    sol = pd.DataFrame(sol)
    sol.to_csv("output/frist_try_MLP.csv", index=False, header=["Id", "y"])
    return

y_train, x_train = get_train_data("train.h5")
x_test = get_test_data("test.h5")

print("building MLPClassifier\n")
clf = MLPClassifier(hidden_layer_sizes=(3,10), activation="relu",solver="sgd")
print("training\n")
clf.fit(x_train, y_train)
print("tessting\n")
y_test = clf.predict(x_test)
print("writing to output\n")
write_test_to_csv(y_test)


