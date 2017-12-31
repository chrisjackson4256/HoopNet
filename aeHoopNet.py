import datetime as dt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.models import model_from_json
from keras.callbacks import TensorBoard
import joblib
import warnings
warnings.filterwarnings('ignore')

# hyper-parameters
num_classes = 2

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

ae_split = int(0.7 * len(X))
X_train_ae = X[ae_split:]
X = X[:ae_split]
y = y[:ae_split]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

print("============")
print("Data details")
print("============")
print("Shape of the AE training data: ", X_train_ae.shape)
print("Shape of the training data: ", X_train.shape)
print("Shape of the validation data: ", X_test.shape)
print("Number of training targets: ", y_train.shape)
print("Number of validation targets: ", y_test.shape)
print()

print("========================")
print("Training the autoencoder")
print("========================")
print()

# model architecture
input_img = Input(shape=(128,))
encoded = Dense(64, activation='relu')(input_img)
decoded = Dense(128, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='RMSprop', loss='binary_crossentropy')

autoencoder.fit(X_train_ae, X_train_ae,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(X_train, X_train),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# run the training data thru the autoencoder
X_train = autoencoder.predict(X_train)

# train some baseline models
print()
print("====================================")
print("Training a Logistic Regression Model")
print("====================================")
print()
start = dt.datetime.now()
parameters = {'C': [0.1, 1.0, 10.], 
			  'penalty': ['l1', 'l2']}
logreg = LogisticRegression()
clf_lr = GridSearchCV(logreg, parameters, cv=5)
#clf_lr = LogisticRegression(C=0.1, penalty='l1')

clf_lr.fit(X_train, y_train)
joblib.dump(clf_lr, 'logreg_model.pkl')

end = dt.datetime.now()

print()
print("Best Logistic Regression params: ", clf_lr.best_params_)
print()
print("Logistic Regression accuracy: ", clf_lr.score(X_test, y_test))
print()
print("Training time: ", end - start)