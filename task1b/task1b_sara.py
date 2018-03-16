

import pandas as pd
import numpy as np



data = pd.read_csv("train.csv")
data = data.as_matrix()

n = np.shape(data)[0]

X_total = data[0:, 2:]
Y_total = data[0:, 1]

X_total = np.c_[X_total, X_total**2, np.exp(X_total), np.cos(X_total), np.ones(np.alen(X_total))]

#find correct lamda to minimize the error


#do regression with this lamda











