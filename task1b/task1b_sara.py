

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV

hallo = pd.read_csv("train.csv", index_col="Id")
data = hallo.as_matrix()


#n = np.shape(data)[0][float(i) for i in
print(np.linspace(0.000001, 1.0, num=100))
l=np.linspace(0.000000000000000001, 1.0, num=100)


X_total = data[:, 1:]
Y_total = data[:, 0]



########################333 nicu
#X_teq = hallo.iloc[:, 1:]
#Y_teq = hallo.iloc[:, 0]
#
#print(X_total==X_teq)
#print(Y_total==Y_teq)[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0] +
#
#X_teq = pd.concat([X_teq,
#                    X_teq.applymap(lambda x: np.square(x)),
#                    X_teq.applymap(lambda x: np.exp(x)),
#                    X_teq.applymap(lambda x: np.cos(x))],
#                    axis=1)
#X_teq['x21'] = 1
############################3




X_total = np.c_[X_total, X_total**2, np.exp(X_total), np.cos(X_total), np.ones(np.alen(X_total))]




###########3333 nicu
#print(X_total==X_teq)
#
#testclf=RidgeCV(alphas = l, fit_intercept=True, normalize=True)
#testclf.fit(X_teq, Y_teq)
#b=testclf.coef_
#############33





clf = RidgeCV(alphas=l, fit_intercept=True, normalize=True)  # tol = 0.0001
clf.fit(X_total, Y_total)
a = clf.coef_

result = pd.DataFrame(a)
result.to_csv("sara_smallest lambda.csv", index=False, header=False)

###########33 nicu
#testresult csv", index=False, header=False)

###########33 = pd.DataFrame(b)
#testresult.to_csv("task1b_sara_nicu_version", index=False, header=False)
###########3


print(clf.alpha_)





