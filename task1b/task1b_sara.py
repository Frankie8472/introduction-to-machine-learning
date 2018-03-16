

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV

data = pd.read_csv("train.csv")
data = data.as_matrix()
print(type(data))

n = np.shape(data)[0]

X_total = data[0:, 2:]
Y_total = data[0:, 1]

X_total = np.c_[X_total, X_total**2, np.exp(X_total), np.cos(X_total), np.ones(np.alen(X_total))]


clf = RidgeCV(alphas=[1], cv=900, fit_intercept=False)  # tol = 0.0001

clf.fit(X_total, Y_total, None)
a = clf.coef_

result = pd.DataFrame(a)
result.to_csv("task1b_sara_900fold.csv", index=False, header=False)



#find correct lamda to minimize the error


#do regression with this lamda











