# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Importing the dataset
dataset = pd.read_csv('human-activity.csv', sep=';')
X = dataset.iloc[:, 6:18].values
y = dataset.iloc[:, 18].values

# Encoding the Dependent Variable
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


################################################################# XGBoost ###################################################################
from xgboost import XGBClassifier

## -------------------------------------------------------------- CPU
start_train_xgb = time.time()

# Fitting XGBoost to the Training set

XGBclassifier = XGBClassifier(tree_method='hist')
XGBclassifier.fit(X_train, y_train)

training_time_xgb = time.time() - start_train_xgb

start_pred_xgb = time.time()

# Predicting the Test set results
y_pred_xgb = XGBclassifier.predict(X_test)

pred_time_xgb = time.time() - start_pred_xgb

# Applying k-Fold Cross Validation
accuracy_xgb = cross_val_score(estimator = XGBclassifier, X = X_train, y = y_train, cv = 5)
accuracy_xgb.std()
mean_accuracy_xgb = accuracy_xgb.mean()

## --------------------------------------------------------------- GPU
start_train_xgb_gpu = time.time()

# Fitting XGBoost to the Training set
XGBclassifier_gpu = XGBClassifier(n_jobs = 32, tree_method='gpu_hist')
XGBclassifier_gpu.fit(X_train, y_train)

training_time_xgb_gpu = time.time() - start_train_xgb_gpu

start_pred_xgb_gpu = time.time()

# Predicting the Test set results
y_pred_xgb_gpu = XGBclassifier_gpu.predict(X_test)

pred_time_xgb_gpu = time.time() - start_pred_xgb_gpu

# Applying k-Fold Cross Validation
accuracy_xgb_gpu = cross_val_score(estimator = XGBclassifier_gpu, X = X_train, y = y_train, cv = 5)
accuracy_xgb_gpu.std()
mean_accuracy_xgb_gpu = accuracy_xgb_gpu.mean()

#############################################################################################################################################

#################################################################### LightGBM ###############################################################
from lightgbm.sklearn import LGBMClassifier

## ----------------------------------------------------------------  CPU

LGBclassifier = LGBMClassifier(objective = 'multiclass', device = 'cpu', learning_rate = 0.05, metric = 'multi_logloss', max_depth = 50, n_estimators= 800, num_leaves=100)
start_train_lgb = time.time()
LGBclassifier.fit(X_train, y_train)
training_time_lgb = time.time() - start_train_lgb

start_pred_lgb = time.time()

# Predicting the Test set results
y_pred_lgb = LGBclassifier.predict(X_test)

pred_time_lgb = time.time() - start_pred_lgb
        
# Applying k-Fold Cross Validation
accuracy_lgb = cross_val_score(estimator = LGBclassifier, X = X_train, y = y_train, cv = 5)
accuracy_lgb.std()
mean_accuracy_lgb = accuracy_lgb.mean()

## ----------------------------------------------------------------  GPU
LGBclassifier_gpu = LGBMClassifier(objective = 'multiclass', device = 'gpu', learning_rate = 0.05, metric = 'multi_logloss', max_depth = 50, n_estimators= 800, num_leaves=100)
start_train_lgb_gpu = time.time()
LGBclassifier_gpu.fit(X_train, y_train)
training_time_lgb_gpu = time.time() - start_train_lgb_gpu

start_pred_lgb_gpu = time.time()

# Predicting the Test set results
y_pred_lgb_gpu = LGBclassifier_gpu.predict(X_test)

pred_time_lgb_gpu = time.time() - start_pred_lgb_gpu
        
# Applying k-Fold Cross Validation
accuracy_lgb_gpu = cross_val_score(estimator = LGBclassifier_gpu, X = X_train, y = y_train, cv = 5)
accuracy_lgb_gpu.std()
mean_accuracy_lgb_gpu = accuracy_lgb_gpu.mean()


#######################################################################################################################################


############################################################## CatBoost ###########################################################################

## ----------------------------------------------------------- CPU
start_train_cat = time.time()

from catboost import CatBoostClassifier
CBclassifier = CatBoostClassifier(loss_function='MultiClass', task_type='CPU', learning_rate=0.1, depth=10, l2_leaf_reg=1, iterations=500)
CBclassifier.fit(X_train, y_train)

training_time_cat = time.time() - start_train_cat

start_pred_cat = time.time()

# Predicting the Test set results
y_pred_cat = CBclassifier.predict(X_test)

pred_time_cat = time.time() - start_pred_cat

# Applying k-Fold Cross Validation
accuracy_cat = cross_val_score(estimator = CBclassifier, X = X_train, y = y_train, cv = 5)
accuracy_cat.std()
mean_accuracy_cat = accuracy_cat.mean()


## ------------------------------------------------------------------- GPU
start_train_cat_gpu = time.time()

# Fitting Kernel cat to the Training set
CBclassifier_gpu = CatBoostClassifier(loss_function='MultiClass', task_type='CPU', learning_rate=0.1, depth=10, l2_leaf_reg=1, iterations=500)
CBclassifier_gpu.fit(X_train, y_train)

training_time_cat_gpu = time.time() - start_train_cat_gpu

start_pred_cat_gpu = time.time()

# Predicting the Test set results
y_pred_cat_gpu = CBclassifier_gpu.predict(X_test)

pred_time_cat_gpu = time.time() - start_pred_cat_gpu

# Applying k-Fold Cross Validation
accuracy_cat_gpu = cross_val_score(estimator = CBclassifier_gpu, X = X_train, y = y_train, cv = 5)
accuracy_cat_gpu.std()
mean_accuracy_cat_gpu = accuracy_cat_gpu.mean()

#############################################################################################################################################

################################################################### GRAPH #############################################################

## ------------------------------------------------------------- Prediction Time --------------------------------------------------- ##

N = 3
fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars

pred_time = (pred_time_xgb, pred_time_lgb, pred_time_cat)
pred_time_gpu = (pred_time_xgb_gpu, pred_time_lgb_gpu, pred_time_cat_gpu)

p1 = ax.bar(ind, pred_time, width, color='blue')

p2 = ax.bar(ind + width, pred_time_gpu, width, color='red')

ax.set_title('GPU CPU Comparision - Prediction Time')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('XGBoost', 'LightGBM', 'CatBoost'))

ax.legend((p1[0], p2[0]), ('CPU', 'GPU'))
ax.autoscale_view()

plt.show()
## --------------------------------------------------------- Training Time ---------------------------------------------------------- ##
fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars

training_time = (training_time_xgb, training_time_lgb, training_time_cat)
training_time_gpu = (training_time_xgb_gpu, training_time_lgb_gpu, training_time_cat_gpu)

t1 = ax.bar(ind, training_time, width, color='blue')

t2 = ax.bar(ind + width, training_time_gpu, width, color='red')

ax.set_title('GPU CPU Comparision - Training Time')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('XGBoost', 'LightGBM', 'CatBoost'))

ax.legend((t1[0], t2[0]), ('CPU', 'GPU'))
ax.autoscale_view()

plt.show()

## --------------------------------------------------------- Mean Accuracy ---------------------------------------------------------- ##
fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars

mean_accuracy = (mean_accuracy_xgb, mean_accuracy_lgb, mean_accuracy_cat)
mean_accuracy_gpu = (mean_accuracy_xgb_gpu, mean_accuracy_lgb_gpu, mean_accuracy_cat_gpu)

a1 = ax.bar(ind, mean_accuracy, width, color='blue')

a2 = ax.bar(ind + width, mean_accuracy_gpu, width, color='red')

ax.set_title('GPU CPU Comparision - Mean Accuracy')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('XGBoost', 'LightGBM', 'CatBoost'))

ax.legend((a1[0], a2[0]), ('CPU', 'GPU'))
ax.autoscale_view()

plt.show()