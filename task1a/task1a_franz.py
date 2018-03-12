
# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd  # tipp from dimitri (python teacher)
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV

precision = np.float128
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

## TRAIN

# read file
raw_data_train = pd.read_csv("train.csv")

# turn raw data into matrix
data_train = raw_data_train.as_matrix()

# filter matrix
y_train = np.asarray(data_train[:, 1], dtype=precision)
X_train = np.asarray(data_train[:, 2:], dtype=precision)

# Create an SGDClassifier instance which will have methods to do our linear regression fitting by gradient descent
fitter = RidgeCV(alphas=alphas,
                 fit_intercept=True,
                 normalize=False,
                 scoring=None,
                 cv=None,
                 gcv_mode='auto',
                 store_cv_values=True
                 )

# Train
fitter.fit(X_train, y_train, sample_weight=None)
print(fitter.cv_values_)
# Print solution to file
#result = pd.DataFrame(data={"": fitter.cv_values_})
#result.to_csv("sample_franz.csv", index=False)
