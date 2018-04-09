
# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

# Import CSV as pandas dataframe
df = pd.read_csv('train.csv', index_col='Id')
test = pd.read_csv('test.csv', index_col='Id')

# Split up data for training
X_train = df.iloc[:,1:]
y_train = df['y']

# Perform training with Multi-layer Perceptron classifier
clf = MLPClassifier(activation='tanh', alpha=10, max_iter=10000, solver='lbfgs')
clf.fit(X_train, y_train)

# Get results
y_out = clf.predict(test)

# Print solution to file
result = pd.DataFrame(data=y_out, index=test.index.values)
result.to_csv("./output/output_nicu_neuralnet3.csv", index=True, index_label="Id", header=['y'])
