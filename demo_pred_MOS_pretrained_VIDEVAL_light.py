# -*- coding: utf-8 -*-
"""
This script predicts a quality score in [1,5] given a VIDEVAL_light feature 
vector by a pretrained VIDEVAL_light model

Input: 

- feature matrix: 
    eg, features/TEST_VIDEOS_VIDEVAL_light720_6fps_feats.mat

Output: 

- predicted scores: 
    eg, results/TEST_VIDEOS_VIDEVAL_light720_6fps_pred.csv

"""
# Load libraries
from sklearn import model_selection
import os
import warnings
import time
import scipy.io
from sklearn.svm import SVR
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
# ignore all warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# Here starts the main part of the script
#
'''======================== parameters ================================''' 

model_name = 'SVR'
data_name = 'TEST_VIDEOS'
algo_name = 'VIDEVAL_light720_6fps'
mat_file = os.path.join('features', data_name+'_'+algo_name+'_feats.mat')
model_file = os.path.join('model', algo_name+'_trained_svr.pkl')
scaler_file = os.path.join('model', algo_name+'_trained_scaler.pkl')
pars_file = os.path.join('model', algo_name+'_logistic_pars.mat')
result_file = os.path.join('results', data_name+'_'+algo_name+'_pred.csv')

print("Predict quality scores using pretrained {} with {} on dataset {} ...".format(
    algo_name, model_name, data_name))

'''======================== read files =============================== '''
X_mat = scipy.io.loadmat(mat_file)
X = np.asarray(X_mat['feats_mat'], dtype=np.float)
X[np.isnan(X)] = 0
X[np.isinf(X)] = 0

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)
popt = np.asarray(scipy.io.loadmat(pars_file)['popt'][0], dtype=np.float)

X = scaler.transform(X)
y_pred = model.predict(X)

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

y = logistic_func(y_pred, *popt)
print('Predicted MOS in [1,5]:')
print(y)
np.savetxt(result_file, y, delimiter=",")