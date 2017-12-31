import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.models import model_from_json
import warnings
warnings.filterwarnings('ignore')

# hyperparameters
batch_size = 32
num_classes = 2
epochs = 20

# read in the historical data
data_list = []

for year in range(2002, 2019):
	data_temp = pd.read_csv("Data/all_games_" + str(year) + ".csv")
	data_list.append(data_temp)

data2002_18 = pd.concat(data_list)

print()
print("Total number of games in data: ", data2002_18.shape[0])

# shuffle data and normalize all columns
data2002_18 = data2002_18.sample(frac=1.0, random_state=4256)
data2002_18.reset_index(inplace=True, drop=True)
min_max_scaler = MinMaxScaler()
for col in data2002_18.columns.tolist():
    data2002_18[col] = min_max_scaler.fit_transform(data2002_18[col].reshape(-1, 1))
    

# split the data into training and validation sets
X = data2002_18.drop('winner', axis=1)
y = data2002_18['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = X_train.as_matrix().reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.as_matrix().reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print()
print("Shape of the training data: ", X_train.shape)
print("Shape of the validation data: ", X_test.shape)
print("Number of training targets: ", y_train.shape)
print("Number of validation targets: ", y_test.shape)
print()


# design the model's architecture
model = Sequential()
model.add(Conv1D(32, kernel_size=3, padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv1D(32, kernel_size=3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv1D(64, kernel_size=3, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, kernel_size=3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
'''
model = Sequential()
input_shape = (128,)
model.add(Dense(64, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
'''

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop()     #rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True)

# Score trained model.
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# serialize model to JSON
#model_json = model.to_json()
#with open("cnnModel.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("cnnModel.h5")
model.save("cnnModel.h5")
print("Saved model to disk")