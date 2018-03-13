
# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
n = 10
seed = None     # Integer for same output
rmse = []
mean_rmse = []

# Import data
data = pd.read_csv("train.csv")

# Shuffle data
data = shuffle(data, random_state=seed)

# Convert to matrix
data = data.as_matrix()

# Split into chunks
data_set = np.array_split(data, indices_or_sections=n)




# Train
for alpha in alphas:
    clf = Ridge(alpha=alpha, copy_X=True, solver="auto")

    for i in range(0, n):
        y_test = data_set[i][:, 1]
        X_test = data_set[i][:, 2:]

        train_set = np.concatenate(data_set[i:])
        y_train = train_set[:, 1]
        X_train = train_set[:, 2:]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        rmse.append(mean_squared_error(y_test, y_pred) ** 0.5)

    mean_rmse.append(np.mean(rmse))
    rmse = []

# Print solution to file
result = pd.DataFrame(mean_rmse)
result.to_csv("sample_franz.csv", index=False, header=False)

print(mean_rmse)
