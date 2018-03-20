
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


lamda = [0.1, 1.0, 10.0, 100.0, 1000.0]
n = 10
seed = None     # Integer for same output
rmse = []
mean_rmse = []


# Import data
data = pd.read_csv("train.csv")

#convert data to matrix
data = data.as_matrix()

# split the data into 10 different sets
data_split = KFold(n_splits=10, shuffle=False, fit_intercept=False, random_state=None)
data_split.get_n_splits(data)

#construct test sets


#for all lamdas
for l in lamda:

    #construct a solver for every lamda
    clf = Ridge(alpha=l, copy_X=True, solver="auto", tol=0.0001)    # tol = 0.0001

    #go through all training indexes
    for train_index, test_index in data_split.split(data):

        #construct the training data sets
        x_train = data[train_index, 2:]
        y_train = data[train_index, 1]
        #train on sets
        clf.fit(x_train, y_train)

        #construct test sets
        x_test = data[test_index, 2:]
        y_test = data[test_index, 1]

        #construct prediciton for test set based on trained module
        y_pred = clf.predict (x_test)

        #calculate the mean squared error for specific partition
        rmse.append(mean_squared_error(y_test, y_pred) ** 0.5)

    #once all partitions are looked at compute the mean
    mean_rmse.append(np.mean(rmse))
    #reset rmse to empty list
    rmse=[]



#convert back to output file
result = pd.DataFrame(mean_rmse)
result.to_csv("1a_final_sara_without_fit_intercept", index=False, header=False)

print(mean_rmse)


