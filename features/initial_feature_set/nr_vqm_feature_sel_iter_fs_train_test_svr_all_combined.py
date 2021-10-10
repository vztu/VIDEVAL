# -*- coding: utf-8 -*-
"""
Author: Zhengzhong Tu
"""
# Load libraries
import warnings
import time
import pandas
import math
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
from sklearn.ensemble import RandomForestRegressor
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
# from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
# ignore all warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# Here starts the main part of the script
#
# Video Multi-metric Ensemble   
model_name = 'SVR'
data_name = 'ALL_COMBINED'
algo_name = 'BRISQUE_feat_sel+GMLOG_feat_sel+HIGRADE_grad_feat_sel+FRIQUEE_feat_sel+TLVQM_naiveLCF_feat_sel+TLVQM_naiveHCF_feat_sel'
color_only = True
algo_names = algo_name.split('+')
sel_algo_name = 'iter_sfs_svr'
n_features = 100
i_repeats = 100

# sel_result_file = './feat_sel_mats/'+data_name+'_'+sel_algo_name+'_sel_results.mat'
sel_result_file = '/home/lhc/Desktop/'+data_name+'_'+sel_algo_name+'_sel_results_1_'+\
    str(n_features)+'_'+str(i_repeats)+'x'+'.mat'
print(sel_result_file)
## read KONVID_1K
data_name = 'KONVID_1K'
# read mos data
csv_file = './mos_feat_files/'+data_name+'_metadata.csv'
try:
    df = pandas.read_csv(csv_file, skiprows=[], header=None)       
except:
    raise Exception('Read csv file error!')
array = df.values  
y1 = array[1:,1]
y1 = np.array(list(y1), dtype=np.float)
# read feature vecs
X1 = []
for k in range(len(algo_names)):
    mat_file = './feat_sel_mats/'+data_name+'_'+algo_names[k]+'_feats.mat'
    X1_mat = scipy.io.loadmat(mat_file)
    X1_mat = np.asarray(X1_mat['feats_mat'], dtype=np.float)
    X1.append(X1_mat)
X1 = np.hstack(X1)
X1[np.isnan(X1)] = 0
X1[np.isinf(X1)] = 0
# apply scaling transform of y1
temp = np.divide(5.0 - y1, 4.0) # convert MOS do distortion
temp = -0.0993 + 1.1241 * temp # apply gain and shift produced by INSLA
temp = 5.0 - 4.0 * temp # convert distortion to MOS
y1 = temp

## read LIVE-VQC
data_name = 'LIVE_VQC'
csv_file = './mos_feat_files/'+data_name+'_metadata.csv'
try:
    df = pandas.read_csv(csv_file, skiprows=[], header=None)
except:
    raise Exception('Read csv file error!')
array = df.values  
y2 = array[1:,1]
y2 = np.array(list(y2), dtype=np.float)
X2 = []
for k in range(len(algo_names)):
    mat_file = './feat_sel_mats/'+data_name+'_'+algo_names[k]+'_feats.mat'
    X2_mat = scipy.io.loadmat(mat_file)
    X2_mat = np.asarray(X2_mat['feats_mat'], dtype=np.float)
    X2.append(X2_mat)
X2 = np.hstack(X2)
X2[np.isnan(X2)] = 0
X2[np.isinf(X2)] = 0
# apply scaling transform of y2
temp = np.divide(100.0 - y2, 100.0) # convert MOS do distortion
temp = 0.0253 + 0.7132 * temp # apply gain and shift produced by INSLA
temp = 5.0 - 4.0 * temp # convert distortion to MOS
y2 = temp

## read YOUTUBE_UGC
data_name = 'YOUTUBE_UGC'
csv_file = './mos_feat_files/'+data_name+'_metadata.csv'
try:
    df = pandas.read_csv(csv_file, skiprows=[], header=None)
except:
    raise Exception('Read csv file error!')
array = df.values  
y3 = array[1:,4]
y3 = np.array(list(y3), dtype=np.float)
X3 = []
for k in range(len(algo_names)):
    mat_file = './feat_sel_mats/'+data_name+'_'+algo_names[k]+'_feats.mat'
    X3_mat = scipy.io.loadmat(mat_file)
    X3_mat = np.asarray(X3_mat['feats_mat'], dtype=np.float)
    X3.append(X3_mat)
X3 = np.hstack(X3)
X3[np.isnan(X3)] = 0
X3[np.isinf(X3)] = 0
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

MOS_predicted_all_repeats_kfeat = []
mask_all_repeats_kfeat = []
scores_all_repeats_kfeat = []
model_params_all_repeats_kfeat = []
PLCC_all_repeats_test_kfeat = []
SRCC_all_repeats_test_kfeat = []
KRCC_all_repeats_test_kfeat = []
RMSE_all_repeats_test_kfeat = []
popt_all_repeats_test_kfeat = []
PLCC_all_repeats_train_kfeat = []
SRCC_all_repeats_train_kfeat = []
KRCC_all_repeats_train_kfeat = []
RMSE_all_repeats_train_kfeat = []
popt_all_repeats_train_kfeat = []

# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.
# 
# C_range = np.logspace(1, 10, 10, base=2)
# gamma_range = np.logspace(-8, 1, 10, base=2)
gamma= 0.1
C= pow(2,6)
# epsilon = 0.3
# params_grid = dict(gamma=gamma_range, C=C_range)
for i in range(1, 1+i_repeats):
    print(i,'th repeated 80-20 hold out validation')
    MOS_predicted_all_repeats = []
    mask_all_repeats = []
    scores_all_repeats = []
    model_params_all_repeats = []
    PLCC_all_repeats_test = []
    SRCC_all_repeats_test = []
    KRCC_all_repeats_test = []
    RMSE_all_repeats_test = []
    popt_all_repeats_test = []
    PLCC_all_repeats_train = []
    SRCC_all_repeats_train = []
    KRCC_all_repeats_train = []
    RMSE_all_repeats_train = []
    popt_all_repeats_train = []

 # Split data to test and validation sets randomly   
    test_size = 0.2
    X_train, X_test, y_train, y_test = \
        model_selection.train_test_split(X, y, test_size=test_size, random_state=math.ceil(8.8*i))

    # select k_best by support vector regressor
    t_start = time.time()

    model = SVR(kernel='rbf', gamma=gamma, C=C)
    feature_selector = SFS(model,
        k_features = n_features, 
        forward=True,
        floating=True,
        verbose=1,
        scoring='neg_mean_squared_error',
        cv=4,
        n_jobs=12)

    features = feature_selector.fit(X_train, y_train)

    print(' -- ' + str(time.time()-t_start) + ' seconds elapsed in SFFS...\n\n')


    for k_features in range(1,n_features+1):
        mask_vec = np.zeros(X.shape[1])
        indices = list(features.subsets_.get(k_features)['feature_idx'])
        mask_vec[indices] = 1
        mask_all_repeats.append(mask_vec.tolist())


    for k_features in range(1,n_features+1):
        t0 = time.time()
        # parameters for each hold out
        mask_vec = mask_all_repeats[k_features-1]
        mask = [i for i,e in enumerate(mask_vec) if e != 0]
        print(mask)
        # drop unselected features
        X_train_reduced = X_train[:,mask]
        X_test_reduced = X_test[:,mask]
        # model_params_all.append((C, gamma))
        # Regression training here. You can use any regression model, here
        # we show the models used in the original work (SVR and RFR)
        if algo_name == 'CORNIA10K' or algo_name == 'HOSA':
            model = SVR(kernel='linear', gamma=gamma, C=C)
        else:
            model = SVR(kernel='rbf', gamma=gamma, C=C)
        # Standard min-max normalization of features
        scaler = preprocessing.MinMaxScaler().fit(X_train_reduced)
        X_train_reduced = scaler.transform(X_train_reduced)  

        # Fit training set to the regression model
        model.fit(X_train_reduced, y_train)

        # Apply scaling 
        X_test_reduced = scaler.transform(X_test_reduced)
    
        # Predict MOS for the validation set
        y_train_pred = model.predict(X_train_reduced)
        y_test_pred = model.predict(X_test_reduced)
        # kf_y_pred_all[param_valid_idx] = y_param_valid_pred

        def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
            logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
            yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
            return yhat
        y_train_pred = np.array(list(y_train_pred), dtype=np.float)
        y_test_pred = np.array(list(y_test_pred), dtype=np.float)
        try:
            # logistic regression
            beta = [np.max(y_train), np.min(y_train), np.mean(y_train_pred), 0.5]
            popt, _ = curve_fit(logistic_func, y_train_pred, \
                y_train, p0=beta, maxfev=100000000)
            y_train_pred_logistic = logistic_func(y_train_pred, *popt)
            # logistic regression
            beta = [np.max(y_test), np.min(y_test), np.mean(y_test_pred), 0.5]
            popt, _ = curve_fit(logistic_func, y_test_pred, \
                y_test, p0=beta, maxfev=100000000)
            y_test_pred_logistic = logistic_func(y_test_pred, *popt)
        except:
            raise Exception('Fitting logistic function time-out!!')

        plcc_test_opt = scipy.stats.pearsonr(y_test, y_test_pred_logistic)[0]
        rmse_test_opt = np.sqrt(mean_squared_error(y_test, y_test_pred_logistic))
        srcc_test_opt = scipy.stats.spearmanr(y_test, y_test_pred)[0]
        krcc_test_opt = scipy.stats.kendalltau(y_test, y_test_pred)[0]

        plcc_train_opt = scipy.stats.pearsonr(y_train, y_train_pred_logistic)[0]
        rmse_train_opt = np.sqrt(mean_squared_error(y_train, y_train_pred_logistic))
        srcc_train_opt = scipy.stats.spearmanr(y_train, y_train_pred)[0]
        krcc_train_opt = scipy.stats.kendalltau(y_train, y_train_pred)[0]

        model_params_all_repeats.append((C, gamma))
        SRCC_all_repeats_test.append(srcc_test_opt)
        KRCC_all_repeats_test.append(krcc_test_opt)
        PLCC_all_repeats_test.append(plcc_test_opt)
        RMSE_all_repeats_test.append(rmse_test_opt)
        SRCC_all_repeats_train.append(srcc_train_opt)
        KRCC_all_repeats_train.append(krcc_train_opt)
        PLCC_all_repeats_train.append(plcc_train_opt)
        RMSE_all_repeats_train.append(rmse_train_opt)

        # plot figs
        print('======================================================')
        print('Results')
        print('SRCC_train: ', srcc_train_opt)
        print('KRCC_train: ', krcc_train_opt)
        print('PLCC_train: ', plcc_train_opt)
        print('RMSE_train: ', rmse_train_opt)
        print('======================================================')
        print('SRCC_test: ', srcc_test_opt)
        print('KRCC_test: ', krcc_test_opt)
        print('PLCC_test: ', plcc_test_opt)
        print('RMSE_test: ', rmse_test_opt)
        print('MODEL: ', (C, gamma))
        print('======================================================')

        print(' -- ' + str(time.time()-t0) + ' seconds elapsed...\n\n')

    mask_all_repeats_kfeat.append(mask_all_repeats)
    scores_all_repeats_kfeat.append(scores_all_repeats)
    model_params_all_repeats_kfeat.append(model_params_all_repeats)
    SRCC_all_repeats_train_kfeat.append(SRCC_all_repeats_train)
    KRCC_all_repeats_train_kfeat.append(KRCC_all_repeats_train)
    PLCC_all_repeats_train_kfeat.append(PLCC_all_repeats_train)
    RMSE_all_repeats_train_kfeat.append(RMSE_all_repeats_train)
    SRCC_all_repeats_test_kfeat.append(SRCC_all_repeats_test)
    KRCC_all_repeats_test_kfeat.append(KRCC_all_repeats_test)
    PLCC_all_repeats_test_kfeat.append(PLCC_all_repeats_test)
    RMSE_all_repeats_test_kfeat.append(RMSE_all_repeats_test)

print(np.asarray(KRCC_all_repeats_test_kfeat).shape)
#mask_all_repeats_kfeat=np.array([[[coord for coord in xk] for xk in xj] for xj in mask_all_repeats_kfeat], ndmin=3) #this case for N=3

#================================================================================
# save figures
scipy.io.savemat(sel_result_file, \
    mdict={'mask_all_repeats_kfeat': np.asarray(mask_all_repeats_kfeat, dtype=np.float), \
        'scores_all_repeats_kfeat': np.asarray(scores_all_repeats_kfeat,dtype=np.float), \
        'model_params_all_repeats_kfeat': np.asarray(model_params_all_repeats_kfeat,dtype=np.float), \
        'SRCC_all_repeats_train_kfeat': np.asarray(SRCC_all_repeats_train_kfeat,dtype=np.float), \
        'KRCC_all_repeats_train_kfeat': np.asarray(KRCC_all_repeats_train_kfeat,dtype=np.float), \
        'PLCC_all_repeats_train_kfeat': np.asarray(PLCC_all_repeats_train_kfeat,dtype=np.float), \
        'RMSE_all_repeats_train_kfeat': np.asarray(RMSE_all_repeats_train_kfeat,dtype=np.float), \
        'SRCC_all_repeats_test_kfeat': np.asarray(SRCC_all_repeats_test_kfeat,dtype=np.float),\
        'KRCC_all_repeats_test_kfeat': np.asarray(KRCC_all_repeats_test_kfeat,dtype=np.float),\
        'PLCC_all_repeats_test_kfeat': np.asarray(PLCC_all_repeats_test_kfeat,dtype=np.float),\
        'RMSE_all_repeats_test_kfeat': np.asarray(RMSE_all_repeats_test_kfeat,dtype=np.float),\
    })
