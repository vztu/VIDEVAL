# %matplotlib inline
"""
Evaluate a BVQA model/features on a given dataset trained with support 
vector machine trained for 100 iterations of train-test (80%-20%) split.
SVR hyperparameters were selected by a grid-search tuning on a random 20% 
of the training set.

Input: 

- feature matrices: 
    eg, features/KONVID_1K_VIDEVAL_feats.mat,
        features/LIVE_VQC_VIDEVAL_feats.mat,
        features/YOUTUBE_UGC_VIDEVAL_feats.mat
- MOS files: 
    eg, features/KONVID_1K_metadata.csv,
        features/LIVE_VQC_metadata.csv,
        features/YOUTUBE_UGC_metadata.csv

Output: 

- evaluation results: eg, results/ALL_COMBINED_VIDEVAL_corr.mat

"""
# Load libraries
from sklearn import model_selection
import os
import warnings
import time
import pandas
import math
import random as rnd
import scipy.stats
import scipy.io
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
# ignore all warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# Here starts the main part of the script
#
'''======================== parameters ================================''' 

model_name = 'SVR' # regression model
data_name = 'ALL_COMBINED' # dataset name
algo_name = 'VIDEVAL' # evaluated model
color_only = True # if True, YouTube-UGCc dataset; if False, YouTube-UGC
use_inlsa = True # if True, apply INLSA calibration of MOS.
result_file = os.path.join('results', data_name+'_'+algo_name+'_corr.mat')

print("Evaluating algorithm {} with {} on dataset {} ...".format(algo_name, 
    model_name, data_name))

'''======================== read files =============================== '''
## read KONVID_1K
data_name = 'KONVID_1K'
csv_file = os.path.join('features', data_name+'_metadata.csv')
mat_file = os.path.join('features', data_name+'_'+algo_name+'_feats.mat')
df = pandas.read_csv(csv_file, skiprows=[], header=None)
array = df.values
y1 = array[1:,1]
y1 = np.array(list(y1), dtype=np.float)
X_mat = scipy.io.loadmat(mat_file)
X1 = np.asarray(X_mat['feats_mat'], dtype=np.float)
X1[np.isnan(X1)] = 0
X1[np.isinf(X1)] = 0
if use_inlsa:
    # apply scaling transform
    temp = np.divide(5.0 - y1, 4.0) # convert MOS do distortion
    temp = -0.0993 + 1.1241 * temp # apply gain and shift produced by INSLA
    temp = 5.0 - 4.0 * temp # convert distortion to MOS
    y1 = temp

## read LIVE-VQC
data_name = 'LIVE_VQC'
csv_file = os.path.join('features', data_name+'_metadata.csv')
mat_file = os.path.join('features', data_name+'_'+algo_name+'_feats.mat')
df = pandas.read_csv(csv_file, skiprows=[], header=None)
array = df.values  
y2 = array[1:,1]
y2 = np.array(list(y2), dtype=np.float)
X_mat = scipy.io.loadmat(mat_file)
X2 = np.asarray(X_mat['feats_mat'], dtype=np.float)
X2[np.isnan(X2)] = 0
X2[np.isinf(X2)] = 0

if use_inlsa:
    # apply scaling transform
    temp = np.divide(100.0 - y2, 100.0) # convert MOS do distortion
    temp = 0.0253 + 0.7132 * temp # apply gain and shift produced by INSLA
    temp = 5.0 - 4.0 * temp # convert distortion to MOS
    y2 = temp
else:
    # apply linear scaling
    y2 = np.divide(y2, 100.0) * 4.0 + 1.0

## read YOUTUBE_UGC
data_name = 'YOUTUBE_UGC'
csv_file = os.path.join('features', data_name+'_metadata.csv')
mat_file = os.path.join('features', data_name+'_'+algo_name+'_feats.mat')
df = pandas.read_csv(csv_file, skiprows=[], header=None)
array = df.values  
y3 = array[1:,4]
y3 = np.array(list(y3), dtype=np.float)
X_mat = scipy.io.loadmat(mat_file)
X3 = np.asarray(X_mat['feats_mat'], dtype=np.float)
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
# concat three datasets
X = np.vstack((X1, X2, X3))
y = np.vstack((y1.reshape(-1,1), y2.reshape(-1,1), y3.reshape(-1,1))).squeeze()


'''======================== Main Body ===========================''' 

model_params_all_repeats = []
PLCC_all_repeats_test = []
SRCC_all_repeats_test = []
KRCC_all_repeats_test = []
RMSE_all_repeats_test = []
PLCC_all_repeats_train = []
SRCC_all_repeats_train = []
KRCC_all_repeats_train = []
RMSE_all_repeats_train = []
# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.
# 
if algo_name == 'CORNIA10K' or algo_name == 'HOSA':
    C_range = [0.1, 1, 10]
    gamma_range = [0.01, 0.1, 1]
else:
    C_range = np.logspace(1, 10, 10, base=2)
    gamma_range = np.logspace(-8, 1, 10, base=2)
params_grid = dict(gamma=gamma_range, C=C_range)

# 100 random splits
for i in range(1,101):
    print(i, 'th repeated 80-20 hold out test')
    t0 = time.time()
    # parameters for each hold out
    model_params_all = []
    PLCC_all_train = []
    SRCC_all_train = []
    KRCC_all_train = []
    RMSE_all_train = []
    PLCC_all_test = []
    SRCC_all_test = []
    KRCC_all_test = []
    RMSE_all_test = []

    # Split data to test and validation sets randomly   
    test_size = 0.2
    X_train, X_test, y_train, y_test = \
        model_selection.train_test_split(X, y, test_size=test_size, random_state=math.ceil(8.8*i))
    Iter = 0
    # SVR grid search in the TRAINING SET ONLY 
    validation_size = 0.2
    X_param_train, X_param_valid, y_param_train, y_param_valid = \
        model_selection.train_test_split(X_train, y_train, test_size=validation_size, random_state=math.ceil(6.6*i))
    # grid search
    for C in C_range:
        for gamma in gamma_range:
            model_params_all.append((C, gamma))
            if algo_name == 'CORNIA10K' or \
                algo_name == 'HOSA':
                model = SVR(kernel='linear', gamma=gamma, C=C)
            else:
                model = SVR(kernel='rbf', gamma=gamma, C=C)
            # Standard min-max normalization of features
            scaler = MinMaxScaler().fit(X_param_train)
            X_param_train = scaler.transform(X_param_train)  

            # Fit training set to the regression model
            model.fit(X_param_train, y_param_train)

            # Apply scaling 
            X_param_valid = scaler.transform(X_param_valid)
        
            # Predict MOS for the validation set
            y_param_valid_pred = model.predict(X_param_valid)
            y_param_train_pred = model.predict(X_param_train)

            # define 4-parameter logistic regression
            def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
                logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
                yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
                return yhat
            y_param_valid = np.array(list(y_param_valid), dtype=np.float)
            y_param_train = np.array(list(y_param_train), dtype=np.float)
            try:
                # logistic regression
                beta = [np.max(y_param_valid), np.min(y_param_valid), np.mean(y_param_valid_pred), 0.5]
                popt, _ = curve_fit(logistic_func, y_param_valid_pred, \
                    y_param_valid, p0=beta, maxfev=100000000)
                y_param_valid_pred_logistic = logistic_func(y_param_valid_pred, *popt)
                # logistic regression
                beta = [np.max(y_param_train), np.min(y_param_train), np.mean(y_param_train_pred), 0.5]
                popt, _ = curve_fit(logistic_func, y_param_valid_pred, \
                    y_param_valid, p0=beta, maxfev=100000000)
                y_param_train_pred_logistic = logistic_func(y_param_train_pred, *popt)
            except:
                raise Exception('Fitting logistic function time-out!!')
            plcc_valid_tmp = scipy.stats.pearsonr(y_param_valid, y_param_valid_pred_logistic)[0]
            rmse_valid_tmp = np.sqrt(mean_squared_error(y_param_valid, y_param_valid_pred_logistic))
            srcc_valid_tmp = scipy.stats.spearmanr(y_param_valid, y_param_valid_pred)[0]
            krcc_valid_tmp = scipy.stats.kendalltau(y_param_valid, y_param_valid_pred)[0]
            plcc_train_tmp = scipy.stats.pearsonr(y_param_train, y_param_train_pred_logistic)[0]
            rmse_train_tmp = np.sqrt(mean_squared_error(y_param_train, y_param_train_pred_logistic))
            srcc_train_tmp = scipy.stats.spearmanr(y_param_train, y_param_train_pred)[0]
            try:
                krcc_train_tmp = scipy.stats.kendalltau(y_param_train, y_param_train_pred)[0]
            except:
                krcc_train_tmp = scipy.stats.kendalltau(y_param_train, y_param_train_pred, method='asymptotic')[0]
            # save results
            PLCC_all_test.append(plcc_valid_tmp)
            RMSE_all_test.append(rmse_valid_tmp)
            SRCC_all_test.append(srcc_valid_tmp)
            KRCC_all_test.append(krcc_valid_tmp)
            PLCC_all_train.append(plcc_train_tmp)
            RMSE_all_train.append(rmse_train_tmp)
            SRCC_all_train.append(srcc_train_tmp)
            KRCC_all_train.append(krcc_train_tmp)

    # using the best chosen parameters to test on testing set
    param_idx = np.argmin(np.asarray(RMSE_all_test, dtype=np.float))
    C_opt, gamma_opt = model_params_all[param_idx]
    if algo_name == 'CORNIA10K' or \
        algo_name == 'HOSA':
        model = SVR(kernel='linear', gamma=gamma_opt, C=C_opt)
    else:
        model = SVR(kernel='rbf', gamma=gamma_opt, C=C_opt)
    # Standard min-max normalization of features
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)  
    
    # Fit training set to the regression model
    model.fit(X_train, y_train)

    # Apply scaling 
    X_test = scaler.transform(X_test)
    
    # Predict MOS for the validation set
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    y_test = np.array(list(y_test), dtype=np.float)
    y_train = np.array(list(y_train), dtype=np.float)
    try:
        # logistic regression
        beta = [np.max(y_test), np.min(y_test), np.mean(y_test_pred), 0.5]
        popt, _ = curve_fit(logistic_func, y_test_pred, \
            y_test, p0=beta, maxfev=100000000)
        y_test_pred_logistic = logistic_func(y_test_pred, *popt)
        # logistic regression
        beta = [np.max(y_train), np.min(y_train), np.mean(y_train_pred), 0.5]
        popt, _ = curve_fit(logistic_func, y_train_pred, \
            y_train, p0=beta, maxfev=100000000)
        y_train_pred_logistic = logistic_func(y_train_pred, *popt)
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

    model_params_all_repeats.append((C_opt, gamma_opt))
    SRCC_all_repeats_test.append(srcc_test_opt)
    KRCC_all_repeats_test.append(krcc_test_opt)
    PLCC_all_repeats_test.append(plcc_test_opt)
    RMSE_all_repeats_test.append(rmse_test_opt)
    SRCC_all_repeats_train.append(srcc_train_opt)
    KRCC_all_repeats_train.append(krcc_train_opt)
    PLCC_all_repeats_train.append(plcc_train_opt)
    RMSE_all_repeats_train.append(rmse_train_opt)

    # print results for each iteration
    print('======================================================')
    print('Best results in CV grid search within one split')
    print('SRCC_train: ', srcc_train_opt)
    print('KRCC_train: ', krcc_train_opt)
    print('PLCC_train: ', plcc_train_opt)
    print('RMSE_train: ', rmse_train_opt)
    print('======================================================')
    print('SRCC_test: ', srcc_test_opt)
    print('KRCC_test: ', krcc_test_opt)
    print('PLCC_test: ', plcc_test_opt)
    print('RMSE_test: ', rmse_test_opt)
    print('MODEL: ', (C_opt, gamma_opt))
    print('======================================================')

    print(' -- ' + str(time.time()-t0) + ' seconds elapsed...\n\n')

print('\n\n')
print('======================================================')
print('Average training results among all repeated 80-20 holdouts:')
print('SRCC: ',np.median(SRCC_all_repeats_train),'( std:',np.std(SRCC_all_repeats_train),')')
print('KRCC: ',np.median(KRCC_all_repeats_train),'( std:',np.std(KRCC_all_repeats_train),')')
print('PLCC: ',np.median(PLCC_all_repeats_train),'( std:',np.std(PLCC_all_repeats_train),')')
print('RMSE: ',np.median(RMSE_all_repeats_train),'( std:',np.std(RMSE_all_repeats_train),')')
print('======================================================')
print('Average testing results among all repeated 80-20 holdouts:')
print('SRCC: ',np.median(SRCC_all_repeats_test),'( std:',np.std(SRCC_all_repeats_test),')')
print('KRCC: ',np.median(KRCC_all_repeats_test),'( std:',np.std(KRCC_all_repeats_test),')')
print('PLCC: ',np.median(PLCC_all_repeats_test),'( std:',np.std(PLCC_all_repeats_test),')')
print('RMSE: ',np.median(RMSE_all_repeats_test),'( std:',np.std(RMSE_all_repeats_test),')')
print('======================================================')
print('\n\n')

#================================================================================
# save mats
scipy.io.savemat(result_file, \
    mdict={'SRCC_train': np.asarray(SRCC_all_repeats_train,dtype=np.float), \
        'KRCC_train': np.asarray(KRCC_all_repeats_train,dtype=np.float), \
        'PLCC_train': np.asarray(PLCC_all_repeats_train,dtype=np.float), \
        'RMSE_train': np.asarray(RMSE_all_repeats_train,dtype=np.float), \
        'SRCC_test': np.asarray(SRCC_all_repeats_test,dtype=np.float), \
        'KRCC_test': np.asarray(KRCC_all_repeats_test,dtype=np.float), \
        'PLCC_test': np.asarray(PLCC_all_repeats_test,dtype=np.float), \
        'RMSE_test': np.asarray(RMSE_all_repeats_test,dtype=np.float),\
    })