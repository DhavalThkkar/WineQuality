#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 16:23:05 2017

@author: thakkar_
"""

class AllInOne:
    acc = []
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        
    
    def logistic(self):
        """
        Logistic Regression fast forward implementation.
        """
        global acc
        # Fit the Logistic regression model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        a = self.metrics(y_test,y_pred)
        acc = np.append(acc, a)
        
    def knn(self, n):
        """
        K Nearest Neighbours Implementation
        \nn = Number of Neighbors
        The number of neighbors must be specified before executing
        """
        global acc
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=n)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        a = self.metrics(y_test,y_pred)
        acc = np.append(acc, a)
        
    def metrics(self, a, b):
        """Display the accuracy and other metrics"""
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        print('The accuracy is {0}'.format(accuracy_score(a, b)))
        accuracyscore = accuracy_score(a, b)
        return accuracyscore
        
    def compare(self):
        pass
        
test = AllInOne(X_train, y_train, X_test, y_test)
test.logistic()
test.knn(6)
test.compare()