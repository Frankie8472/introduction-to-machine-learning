import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV

index = pd.read_csv()
index = pd.as_matrix(index)[0,:]

train = pd.read_csv("train.csv", Index=False)
train = pd.as_matrix(train)

test = pd.read_csv("test.csv", Index=False)
test = pd.as_matrix(test)

x_train = train[1:, :]
y_train = train[0, :]

x_test = test

logistic = LogisticRegressionCV(C = e5)
logistic.fit(x_train, y_train)
y_pred = logistic.predict(x_test)

result =
result [0,:] = index
result [1,:] = y_pred