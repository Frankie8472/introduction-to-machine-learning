
# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

# Import CSV as pandas dataframe
df = pd.read_csv('train.csv', index_col='Id')
test = pd.read_csv('test.csv', index_col='Id')

# Splitt up data for training
X_train = df.iloc[:,1:]
y_train = df['y']

# Perform training with one-vs-one multiclass strategy
ovoc = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)

# Get results
y_out = ovoc.predict(test)

# Print solution to file
result = pd.DataFrame(data=y_out, index=test.index.values)
result.to_csv("./output/output_nicu.csv", index=True, index_label="Id", header=['y'])
