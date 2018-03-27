

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV

hallo = pd.read_csv("train.csv", index_col="Id")
data = hallo.as_matrix()


# n = np.shape(data)[0][float(i) for i in
l = np.linspace(0.00001, 1.0, num=100).tolist() + [float(i) for i in list(range(1, 100))]+ [float(i) for i in list(range(600, 1000))]


X_total = data[:, 1:]
Y_total = data[:, 0]



########################333 nicu
#X_teq = hallo.iloc[:, 1:]
#Y_teq = hallo.iloc[:, 0]
#print(clf.alpha_)

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





clf = RidgeCV(alphas=l, fit_intercept=False, normalize=True)  # tol = 0.0001, note: fit_intercept is defautl True
clf.fit(X_total, Y_total)
a = clf.coef_
print(clf.alpha_)
result = pd.DataFrame(a)
result.to_csv("sara_1b_fit_intercept=False.csv", index=False, header=False)




###########33 nicu
#testresult csv", index=False, header=False)print(clf.alpha_)


###########33 = pd.DataFrame(b)
#testresult.to_csv("task1b_sara_nicu_version", index=False, header=False)
###########3


print(clf.alpha_)





