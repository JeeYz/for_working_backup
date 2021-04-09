import os
import sys
import random
import matplotlib.pyplot as plt

import numpy as np
import time
from scipy.io import wavfile
from check_files import createFolder, make_data_list

from sklearn.preprocessing import Normalizer

temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)

from analysis_signal import draw_single_graph, signal_trigger, util_module
from python_speech_features import logfbank
from analysis_signal.util_module import new_minmax_normal

mod_full_data_files_name = 'D:\\mod_full_data_files_list.txt'

aug_train_data_path = 'D:\\aug_train_data.npz'
aug_test_data_path = 'D:\\aug_test_data.npz'

norm_train_data_path = 'D:\\aug_norm_train_data.npz'
norm_test_data_path = 'D:\\aug_norm_test_data.npz'


##
def main():

    normalize_data()

    return


##
def normalize_data():
    samplerate = 16000

    fwb_train = open(norm_train_data_path, 'wb')
    fwb_test = open(norm_test_data_path, 'wb')

    load_train_data = np.load(aug_train_data_path, allow_pickle=True)
    load_test_data = np.load(aug_test_data_path, allow_pickle=True)

    train_data = load_train_data['data']
    train_label = load_train_data['label']

    test_data = load_test_data['data']
    test_label = load_test_data['label']

    new_train_data = list()
    for i,one in enumerate(train_data):
        # transformer = Normalizer(norm='l1').fit(one)
        # data = transformer.transform(one)
        # new_train_data.append(data)
        new_train_data.append(new_minmax_normal(one))
        print('one data complete...', i)

    train_data = np.asarray(new_train_data)

    new_test_data = list()
    for i,one in enumerate(test_data):
        # transformer = Normalizer(norm='l1').fit(one)
        # data = transformer.transform(one)
        # new_test_data.append(data)
        new_test_data.append(new_minmax_normal(one))
        print('one data complete...', i)

    test_data = np.asarray(new_test_data)

    print(np.max(train_data))
    print(np.min(train_data))

    np.savez_compressed(fwb_train, label=train_label, data=train_data, rate=samplerate)
    np.savez_compressed(fwb_test, label=test_label, data=test_data, rate=samplerate)

    return









## endl

if __name__ == '__main__':
    main()
