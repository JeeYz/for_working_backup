# -*- coding: utf-8 -*-

#%% import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing

#%% standardization



#%% new min max normal
def new_minmax_normal(input_data):
    temp = list()
    for one in input_data:
        one = np.array(one, dtype='float32')
        res = (one - np.min(one))/(np.max(one) - np.min(one))
        # res = res*2 - 1.0
        temp.append(res)
    
    res_data = np.array(temp)
    
    return res_data