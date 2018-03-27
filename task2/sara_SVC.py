from sklearn import svm
import numpy as np
import pandas as pd


train = pd.read_csv("train.csv", index_col="Id")
train = train.as_matrix()
x_train = train[:, 1:]
y_train = train[:, 0]

test = pd.read_csv("test.csv")
test = test.as_matrix()
index_test = test[:, 0]
x_test = test[:, 1:]

solver = svm.SVC(gamma=0.001, C=100.)
solver.fit(x_train, y_train)
result = solver.predict(y_train)

result = np.c_[index_test, result]
solution = pd.DataFrame(result)
solution.to_csv("sara_svc.csv", index=False, head=["Id", "y"])

