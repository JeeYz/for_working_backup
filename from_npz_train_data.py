# -*- coding: utf-8 -*-

#%% explained
"""
class : make train data
methods :
    1. write raw train data. e.g. mfcc files -> *.npz
    2. make raw signal data
    3. modification labels with mode
"""

#%% declaration
import sys
sys.path.append("C:\\Users\\jyback_pnc\\Desktop\\code\\girl_from_ipanema")

import numpy as np
from python_speech_features import mfcc
from python_speech_features import logfbank
from preprocessing_for_data import new_minmax_normal


#%%
class make_train_feature:

    def __init__(self, **kwarg):
        if "return_path" in kwarg.keys():
            self.return_path = kwarg['return_path']


#%% normalization train data
    def normalize_train_data(self, **kwarg):
        if "origin_filename" in kwarg.keys():
            origin_filename = kwarg["origin_filename"]
        if "norm_filename" in kwarg.keys():
            norm_filename = kwarg["norm_filename"]

        for ori_f, norm_f in zip(origin_filename, norm_filename):
            ori_f = self.return_path+'\\'+ori_f

            loaded_data = np.load(ori_f, allow_pickle=True)

            label = loaded_data['label']
            origin_data = loaded_data['data']
            sample_rate = loaded_data['rate']

            mod_data = new_minmax_normal(origin_data)

            fwb = open(norm_f, 'wb')
            np.savez_compressed(fwb, label=label, data=mod_data, rate=sample_rate)
            fwb.close()

        return


#%% write log filter bank data
    def make_train_logfb(self, **kwarg):

        if "raw_filename" in kwarg.keys():
            raw_filename = kwarg["raw_filename"]
        if "logfb_filename" in kwarg.keys():
            logfb_filename = kwarg["logfb_filename"]

        num = 0
        for raw_f, logfb_f in zip(raw_filename, logfb_filename):
            raw_f = self.return_path+'\\'+raw_f

            loaded_data = np.load(raw_f, allow_pickle=True)

            label = loaded_data['label']
            raw_signal = loaded_data['data']
            sample_rate = loaded_data['rate']

            logfb_list = list()

            for sig, rate in zip(raw_signal, sample_rate):
                logfb_feat = logfbank(sig, rate)
                logfb_list.append(logfb_feat)
                num += 1
                print(num)

            logfb_data = np.asarray(logfb_list)

            fwb = open(logfb_f, 'wb')
            np.savez_compressed(fwb, label=label, data=logfb_data, rate=sample_rate)
            fwb.close()

        return


#%% write mfcc files
    def make_train_mfcc(self, **kwarg):

        if "raw_filename" in kwarg.keys():
            raw_filename = kwarg["raw_filename"]
        if "mfcc_filename" in kwarg.keys():
            mfcc_filename = kwarg["mfcc_filename"]
        if "number_of_ceps" in kwarg.keys():
            num_ceps = kwarg['number_of_ceps']


        for raw_f, mfcc_f in zip(raw_filename, mfcc_filename):
            raw_f = self.return_path+'\\'+raw_f

            loaded_data = np.load(raw_f, allow_pickle=True)

            label = loaded_data['label']
            raw_signal = loaded_data['data']
            sample_rate = loaded_data['rate']

            mfcc_list = list()

            for sig, rate in zip(raw_signal, sample_rate):
                mfcc_feat = mfcc(sig, rate, numcep=num_ceps)
                mfcc_list.append(mfcc_feat)

            mfcc_data = np.asarray(mfcc_list)

            fwb = open(mfcc_f, 'wb')
            np.savez_compressed(fwb, label=label, data=mfcc_data, rate=sample_rate)
            fwb.close()

        return














#%% __main__
if __name__=="__main__":
    print("hello, world~!!")
