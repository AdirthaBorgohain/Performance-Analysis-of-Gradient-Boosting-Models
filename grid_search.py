# Grid Search for Algorithms

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Importing the dataset
dataset = pd.read_csv('shuttle.csv', header=None)
X = dataset.iloc[:, 0:9].values
y = dataset.iloc[:, 9].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from catboost import CatBoostClassifier
CBclassifier = CatBoostClassifier(loss_function='MultiClass', task_type='CPU')
CBclassifier.fit(X_train, y_train)

from sklearn.model_selection import GridSearchCV

parameters = {'depth': [4, 7, 10],
          'learning_rate' : [0.03, 0.1, 0.15],
         'l2_leaf_reg': [1,4,9],
         'iterations': [300]
         }

grid_search = GridSearchCV(CBclassifier, n_jobs=-1, param_grid=parameters, cv = 3, scoring='accuracy', verbose=3)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
