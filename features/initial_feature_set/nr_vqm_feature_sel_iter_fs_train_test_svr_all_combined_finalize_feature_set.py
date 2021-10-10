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
n_features = 60
i_repeats = 1

#============================= load feature matrix ====================================
# sel_result_file = './feat_sel_mats/'+data_name+'_'+sel_algo_name+'_sel_results.mat'
sel_result_file = '/home/ztu/Desktop/'+data_name+'_'+sel_algo_name+'_sel_results_1_'+\
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
## =====================================================================================


## ================================== Function ==========================================
def feature_selection(iter, X, y):
    print("{} th iterations".format(iter))
    # init time logger
    t0 = time.time()
    # init model for selection
    model = SVR(kernel='rbf', gamma='scale')
    # init feature selector
    feat_selector = SFS(model,
        k_features = n_features,
        forward=True,
        floating=True,
        verbose=1,
        scoring='neg_mean_squared_error',
        cv=20,
        n_jobs=6)
    # fit selector
    features = feat_selector.fit(X, y)
    # extract selected feature indices 
    for k in range(1, n_features+1):
        mask_vec = np.zeros(X.shape[1])
        indices = list(features.subsets_.get(k)['feature_idx'])
        mask_vec[indices] = 1
        mask_vec = mask_vec.tolist()
    print("{} th iterations finished! {} secs elapsed...".format(iter, str(time.time() - t0)))
    # return mask
    return mask_vec

from joblib import Parallel, delayed
import multiprocessing
## ================================== Main Function ====================================
if __name__ == '__main__':

    num_cores = multiprocessing.cpu_count()
    iters = range(1, 1+i_repeats)
    mask_all_repeats_kfeat = Parallel(n_jobs=num_cores)(
        delayed(feature_selection)(i, X, y) for i in iters)
    # print(mask_all_repeats_kfeat)
    # save figures
    scipy.io.savemat(sel_result_file, \
        mdict={'mask_all_repeats_kfeat': np.asarray(mask_all_repeats_kfeat, dtype=np.float)
        })
