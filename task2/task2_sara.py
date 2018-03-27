import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV

index = pd.read_csv("test.csv")
index = index.as_matrix()
print(type(index))
index = index[:, 0]

train = pd.read_csv("train.csv", index_col="Id")
train = train.as_matrix()

test = pd.read_csv("test.csv", index_col="Id")
test = test.as_matrix()

x_train = train[:, 1:]
y_train = train[:, 0]

x_test = test

logistic = LogisticRegressionCV(Cs=[1e5])
logistic.fit(x_train, y_train)
y_pred = logistic.predict(x_test)

print(y_pred[0:10],type(y_pred))
print(index[0:10], type(index))

#pach y pred to index
result = np.c_[index, y_pred]
print(type(result))
print(result[:, :])
#;alskfdj



output = pd.DataFrame(result)
output.to_csv("task2_sara.csv", index=False, header=False)