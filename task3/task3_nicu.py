
# train.csv     - the training set
# test.csv      - the test set (make predictions based on this file)
# sample.csv    - a sample submission file in the correct format
import keras
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten


# Import HDF as pandas dataframe
train = pd.read_hdf('./input/train.h5', 'train')
test = pd.read_hdf('./input/test.h5', 'test')

# Split up data for training and validation
X_train = train.iloc[:,1:]
y_train = train['y']

# Reshape data
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(test, axis=2)

# Define types
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize data to [0,1] range
X_train /= np.max(X_train)
X_test /= np.max(X_test)

# Initiate model and turn training data labels into matrix
model = Sequential()
y_train = keras.utils.to_categorical(y_train, num_classes=5)

# Stack desired layers
model.add(Conv1D(filters=32, kernel_size=5, input_shape=(100, 1), strides=1, activation='relu'))
model.add(Conv1D(filters=32, kernel_size=5, strides=1, activation='tanh'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Perform training with 20% of the data used for validation, stops early if val_loss doesn't decrease
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='min')
model.fit(X_train, y_train, validation_split=0.2, epochs=500, batch_size=100, verbose=1, callbacks=[earlystop])

# Get result
y_out = model.predict_classes(X_test, batch_size=100)

# Print solution to file
result = pd.DataFrame(data=y_out, index=test.index.values)
result.to_csv("./output/output_nicu.csv", index=True, index_label="Id", header=['y'])
