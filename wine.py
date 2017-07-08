#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 12:54:02 2017

@author: thakkar_
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

red = pd.read_csv('winequality-red.csv', delimiter=';')

# X and Y dataframes
X = red.iloc[:,:11].values
y = red.iloc[:,11:12].values

# X and Y for Support Vector machines
X = red.drop('quality', axis = 1, inplace = False)
y = red['quality']

# Label encoding Y for getting output of Neural network
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y[:, 0] = labelencoder_y.fit_transform(y[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()
y = y[:, 1:]

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Standardize all the Values to feed to the Neural Network
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 600, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate = 0.4))

# Adding the second hidden layer
classifier.add(Dense(units = 600, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate= 0.4))

# Adding the output layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 50, epochs = 1000)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = np.array(y_pred)
y_pred = y_pred.astype(float)

# Support Vector Machine
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# Grid Search
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid_predictions = grid.predict(X_test)

# Evaluation of Performance
from sklearn.metrics import classification_report
print(classification_report(y_test,grid_predictions))