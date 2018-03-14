
# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Set up given constants
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
k = 10 # k-fold cross-validation

# Import CSV as pandas dataframe
df = pd.read_csv('train.csv', index_col='Id')

# Determine number of total items and size of each validation set
n = df.shape[0]
val_size = n // k #daring to do simple integer division, no rounding errors in this case ^^

# Result variables
rmse_sum = 0.0
rmse_avg = []

# Iterate over all alphas
for alpha in alphas:

    clf = Ridge(alpha=alpha)

    for i in range(k):

        # Split up dataframe for k-fold cross validation
        val_set = df.loc[i * val_size : (i+1) * val_size - 1]
        train_set = pd.concat([df.loc[0:i * val_size-1],df.loc[(i+1) * val_size:]])

        # Train on train_set
        y_train = train_set['y']
        X_train = train_set.iloc[:, 1:]
        clf.fit(X_train,y_train)

        # Validate on val_set
        y_val = val_set['y']
        X_val = val_set.iloc[:, 1:]
        y_pred = clf.predict(X_val)

        # Store Root Mean Squared Error
        rmse_sum += mean_squared_error(y_val, y_pred)**0.5

    rmse_avg.append(rmse_sum / k)


# Print solution to file
result = pd.DataFrame(rmse_avg)
result.to_csv("output_nicu.csv", index=False, header=False)
