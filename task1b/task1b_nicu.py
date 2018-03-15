
# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV

# Import CSV as pandas dataframe
df = pd.read_csv('train.csv', index_col='Id')

# Define alphas and k
alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
#k = 10

# Apply feature transformation and set up data for training
X_plain = df.iloc[:,1:]
X_train = pd.concat([X_plain,
                    X_plain.applymap(lambda x: np.square(x)),
                    X_plain.applymap(lambda x: np.exp(x)),
                    X_plain.applymap(lambda x: np.cos(x))],
                    axis=1)
X_train['x21'] = 1
y_train = df['y']

# Perform training with ridge regression and cross-validation
reg = RidgeCV(alphas=alphas, fit_intercept=True, normalize=True)
reg.fit(X_train,y_train)

# Get weights
weights = reg.coef_

# Print solution to file
result = pd.DataFrame(weights)
result.to_csv("output_nicu.csv", index=False, header=False)
