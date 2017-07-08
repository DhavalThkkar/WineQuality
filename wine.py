# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

red = pd.read_csv('winequality-red.csv', delimiter=';')
sns.heatmap(data = red.corr(),cmap = 'viridis', yticklabels=False)

# X and Y dataframes
X = red.iloc[:,:11].values
y = red.iloc[:,11:12].values

# Standardize all the Values to feed to the Neural Network
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)