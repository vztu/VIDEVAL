# -*- coding: utf-8 -*-
"""
This script shows how to apply k-folds train and validate regression model to predict
MOS from the features computed with compute_features_example.m

Author: Zhengzhong Tu
"""
import warnings
import time
import os
# ignore all warnings
warnings.filterwarnings("ignore")
# Load libraries
import pandas
import random as rnd
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.stats
import scipy.io
from scipy.optimize import curve_fit
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

# ===========================================================================
# Here starts the main part of the script
#
'''======================== parameters ================================''' 

model_name = 'SVR'
data_name = 'ALL_COMBINED'
algo_name = 'VIDEVAL_light720_6fps'
color_only = False
save_path = 'model'
print("Evaluating algorithm {} with {} on dataset {} ...".format(algo_name, model_name, data_name))

## read KONVID_1K
data_name = 'KONVID_1K'
csv_file = './features/'+data_name+'_metadata.csv'
mat_file = './features/'+data_name+'_'+algo_name+'_feats.mat'
try:
    df = pandas.read_csv(csv_file, skiprows=[], header=None)
except:
    raise Exception('Read csv file error!')
array = df.values
y1 = array[1:,1]
y1 = np.array(list(y1), dtype=np.float)
X_mat = scipy.io.loadmat(mat_file)
X1 = np.asarray(X_mat['feats_mat'], dtype=np.float)
# apply scaling transform
temp = np.divide(5.0 - y1, 4.0) # convert MOS do distortion
temp = -0.0993 + 1.1241 * temp # apply gain and shift produced by INSLA
temp = 5.0 - 4.0 * temp # convert distortion to MOS
y1 = temp

## read LIVE-VQC
data_name = 'LIVE_VQC'
csv_file = './features/'+data_name+'_metadata.csv'
mat_file = './features/'+data_name+'_'+algo_name+'_feats.mat'
try:
    df = pandas.read_csv(csv_file, skiprows=[], header=None)
except:
    raise Exception('Read csv file error!')
array = df.values  
y2 = array[1:,1]
y2 = np.array(list(y2), dtype=np.float)
X_mat = scipy.io.loadmat(mat_file)
X2 = np.asarray(X_mat['feats_mat'], dtype=np.float)
# apply scaling transform
temp = np.divide(100.0 - y2, 100.0) # convert MOS do distortion
temp = 0.0253 + 0.7132 * temp # apply gain and shift produced by INSLA
temp = 5.0 - 4.0 * temp # convert distortion to MOS
y2 = temp
# n, bins, patches = plt.hist(y2, 20)
# plt.show()


## read YOUTUBE_UGC
data_name = 'YOUTUBE_UGC'
csv_file = './features/'+data_name+'_metadata.csv'
mat_file = './features/'+data_name+'_'+algo_name+'_feats.mat'
try:
    df = pandas.read_csv(csv_file, skiprows=[], header=None)
except:
    raise Exception('Read csv file error!')
array = df.values  
y3 = array[1:,4]
y3 = np.array(list(y3), dtype=np.float)
X_mat = scipy.io.loadmat(mat_file)
X3 = np.asarray(X_mat['feats_mat'], dtype=np.float)
#### 57 grayscale videos in YOUTUBE_UGC dataset, we do not consider them for fair comparison ####
if color_only:
    gray_indices = [3,6,10,22,23,46,51,52,68,74,77,99,103,122,136,141,158,173,368,426,467,477,506,563,594,\
    639,654,657,666,670,671,681,690,697,702,703,710,726,736,764,768,777,786,796,977,990,1012,\
    1015,1023,1091,1118,1205,1282,1312,1336,1344,1380]
    gray_indices = [idx - 1 for idx in gray_indices]
    X3 = np.delete(X3, gray_indices, axis=0)
    y3 = np.delete(y3, gray_indices, axis=0)

X = np.vstack((X1, X2, X3))
y = np.vstack((y1.reshape(-1,1), y2.reshape(-1,1), y3.reshape(-1,1))).squeeze()

## preprocessing
from sklearn.impute import SimpleImputer
X[np.isinf(X)] = np.nan
imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
X = imp.transform(X)

## parameter search on k-fold
param_grid = {'C': np.logspace(1, 10, 10, base=2),
              'gamma': np.logspace(-8, 1, 10, base=2)}
grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, n_jobs=4, verbose=2)

## scaler
scaler = preprocessing.MinMaxScaler().fit(X)
X = scaler.transform(X)

# grid search 
grid.fit(X, y)
best_params = grid.best_params_

## finalize SVR model on the combined set
C = best_params['C']
gamma = best_params['gamma']
model = SVR(kernel='rbf', gamma=gamma, C=C)
# Standard min-max normalization of features

# Fit training set to the regression model
model.fit(X, y)

# Apply scaling 
y_pred = model.predict(X)

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat
# logistic regression
try:
    beta = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, pcov = curve_fit(logistic_func, y_pred, \
        y, p0=beta, maxfev=100000000)
except:
    raise Exception('Fitting logistic function time-out!!')
y_pred_logistic = logistic_func(y_pred, *popt)
plcc = scipy.stats.pearsonr(y, y_pred_logistic)[0]
rmse = np.sqrt(mean_squared_error(y, y_pred_logistic))
srcc = scipy.stats.spearmanr(y, y_pred_logistic)[0]
krcc = scipy.stats.kendalltau(y, y_pred_logistic)[0]

# print results for each iteration
print('======================================================')
print('Best results in CV grid search')
print('SRCC: ', srcc)
print('KRCC: ', krcc)
print('PLCC: ', plcc)
print('RMSE: ', rmse)
print('======================================================')

joblib.dump(model, os.path.join(save_path,algo_name+'_trained_svr.pkl'))
joblib.dump(scaler, os.path.join(save_path,algo_name+'_trained_scaler.pkl'))
scipy.io.savemat(os.path.join(save_path,algo_name+'_logistic_pars.mat'), \
    mdict={'popt': np.asarray(popt, dtype=np.float)})
