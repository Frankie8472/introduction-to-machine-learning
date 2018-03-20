

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

#get data ineobject
from sklearn.model_selection import KFold

data = pd.read_csv("train.csv", index_col="Id")
data = data.as_matrix()

#define alphas
alpha = [float(i) for i in list(range(1, 100))]

#read out data
X_total = data[:, 1:]
Y_total = data[:, 0]
msqe = []
per_alpha_msqe = []

#since computing with numpy.array it is possible to directly multiply with funcitons. (elementwise)
#alternative with pandas.core.farame.DataFrame (before conversion to)
#
#pd.concat([X_teq,X_teq.applymap(lambda x: np.square(x))]
#

X_total = np.c_[X_total, X_total ** 2, np.exp(X_total), np.cos(X_total), np.ones(np.alen(X_total))]

data_split = KFold(n_splits=10, shuffle=False,random_state=None)


#test for every alpha the mean squared error made with a 10 fold cross validation
for a in alpha:
    # train the data set for every part and then append the mean squared error
    for train_index, test_index in data_split.split(X_total):

        x_train = X_total[train_index, :]
        y_train = Y_total[train_index]
        x_test = X_total[test_index,:]
        y_test = Y_total[test_index]

        #create learning algorithm
        clf = RidgeCV(alphas=[a], fit_intercept=False)  # tol = 0.0001
        #train on training data
        clf.fit(x_train, y_train)
        #measure goodness
        y_pred = clf.predict(x_test)
        msqe.append(mean_squared_error(y_test, y_pred) ** 0.5)

    #print(a, msqe)
    per_alpha_msqe.append(np.   mean(msqe))
    msque=[]

###
#now find the index of the smalles meansquarederror. that gives the best alpha


print(per_alpha_msqe)

minimum = min(per_alpha_msqe)
best_alpha = alpha[per_alpha_msqe.index(minimum)]

print("The best alpha is", best_alpha)

clf = RidgeCV(alphas=[best_alpha], fit_intercept=False,)
clf.fit(X_total, Y_total)
solution = clf.coef_

result = pd.DataFrame(solution)
result.to_csv("sara_with_perfect_alpha.csv", index=False, header=False)



