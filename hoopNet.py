import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
import warnings
warnings.filterwarnings('ignore')


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

print()
print("Shape of the training data: ", X_train.shape)
print("Shape of the validation data: ", X_test.shape)
print("Number of training targets: ", y_train.shape)
print("Number of validation targets: ", y_test.shape)
print()

# train some baseline models

print("Training a Logistic Regression Model...")
print("---------------------------------------")
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


print()
print("Training a Random Forest Model...")
print("---------------------------------")
start = dt.datetime.now()
parameters = {"n_estimators": [10, 100, 500],
			  "criterion": ['gini', 'entropy'],
			  "min_samples_split": [2, 5, 10],
			  "min_samples_leaf": [1, 2, 5]}
forest = RandomForestClassifier()
clf_rf = GridSearchCV(forest, parameters, cv=5)
#clf_rf = RandomForestClassifier(n_estimators=100, criterion='gini',
#								min_samples_split=5, min_samples_leaf=5)

clf_rf.fit(X_train, y_train)
joblib.dump(clf_rf, 'random_forest_model.pkl')

end = dt.datetime.now()

print()
print("Best Random Forest params: ", clf_rf.best_params_)
print()
print("Random Forest accuracy: ", clf_rf.score(X_test, y_test))
print()
print("Training time: ", end - start)


print()
print("Training a Gradient Boost Model...")
print("---------------------------------")
parameters = {"n_estimators": [10, 100, 500],
			  "loss": ['deviance', 'exponential'],
			  "min_samples_split": [2, 5, 10],
			  "min_samples_leaf": [1, 2, 5]}
gboost = GradientBoostingClassifier()
clf_gb = GridSearchCV(gboost, parameters, cv=5)
#clf_gb = GradientBoostingClassifier(n_estimators=100, loss='deviance',
#									min_samples_split=10, min_samples_leaf=1)

clf_gb.fit(X_train, y_train)
joblib.dump(clf_gb, 'gradient_boost_model.pkl')

end = dt.datetime.now()

print()
print("Best Gradient Boost params: ", clf_gb.best_params_)
print()
print("Gradient Boost accuracy: ", clf_gb.score(X_test, y_test))
print()
print("Training time: ", end - start)

#svm = SVC(probability=True)
#svm.fit(X_train, y_train)
#print("SVM accuracy: ", svm.score(X_test, y_test))
#print()