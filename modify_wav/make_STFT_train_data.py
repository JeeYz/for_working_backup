
import os
import sys
import random
import matplotlib.pyplot as plt
import time

import numpy as np
import time
from scipy.io import wavfile
from check_files import createFolder, make_data_list

from sklearn.preprocessing import Normalizer

from command_mapping import camera_app_mapping

from analysis_signal import draw_single_graph, signal_trigger, util_module
from python_speech_features import logfbank

import librosa

origin_text_file = 'D:\\mod_full_data_files_list.txt'

result_train_data = 'D:\\train_data_for_STFT.npz'

##
data_file_dict = {'file' : list(), 'label' : list()}
def make_data_file_dict():
    global data_file_dict
    count = 0
    with open(origin_text_file, 'r', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            if not line: break
            line = line.split()

            count+=1
            data_file_dict['file'].append(line[0])
            data_file_dict['label'].append(camera_app_mapping[int(line[-1])])
            print('\r{num} th file is complete...'.format(num=count), end='')

    print('\n')

    return


##
rate_list = list()
def make_rate_list():
    global rate_list
    rate_value = 0.1
    num_of_rates = 1

    rate_list.append(1.0)
    for i in range(num_of_rates):
        rate_list.append(1.0+(i+1)*rate_value)
        rate_list.append(1.0-(i+1)*rate_value)

    print(rate_list)
    return


##
full_data = list()
def make_train_npz_file():
    global full_data

    train_data_list = list()
    train_label_list = list()

    fwb_train = open(result_train_data, 'wb')

    count = 0
    max_val = 0

    temp_count = 0
    temp_list = list()

    for i,j in enumerate(data_file_dict['file']):
        sr, data = wavfile.read(j)
        for r in rate_list:
            aug_data = librosa.effects.time_stretch(data, r)
            # if max_val < len(aug_data):
            #     max_val = len(aug_data)

            if len(aug_data) > 32000:
                aug_data = aug_data[:32000]
            elif len(aug_data) < 32000:
                noise_data = np.random.randn(32000-len(aug_data))*0.01
                aug_data = np.concatenate((aug_data, noise_data))

            # if len(aug_data) != 32000:
            #     print(len(aug_data))
            #     time.sleep()

            train_data_list.append(aug_data)
            train_label_list.append(data_file_dict['label'][i])
            # print(data_file_dict['label'][i], j)

            count+=1
            print('\r{num} th data complete...label : {label}'.format(num=count, label=data_file_dict['label'][i]), end='')
        # if i==10: break

    print('\n')
    print('start to save file...')
    train_data_list = np.asarray(train_data_list, dtype='object')
    train_label_list = np.asarray(train_label_list, dtype='object')
    # print(train_data_list)
    # print(train_label_list)
    # train_data_list = np.array(train_data_list, dtype='int16')
    # train_label_list = np.array(train_label_list, dtype='int16')
    np.savez_compressed(fwb_train, label=train_label_list, data=train_data_list, rate=sr)
    print('complete.')

    return


##
if __name__ == '__main__':
    print('hello, world~!!')

    make_data_file_dict()
    make_rate_list()
    make_train_npz_file()


    # temp_file_1 = "D:\\voice_data_backup\\ASR_audio_files\\train\\PNCDB\kdh\\reject\\kdhreject7.wav"
    # temp_file_2 = "D:\\ASR_train_data_mod\\train\\PNCDB_wakeup2\\kdh2\\hipnc2\\kdh2hipnctwo4.wav"
    #
    # sr, data = wavfile.read(temp_file_2)
    # print(np.max(data))
    #
    # # data = np.int16(data)
    # norm_data = (data-np.min(data))/(np.max(data)-np.min(data))*2-1
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(2,1,1)
    # ax2 = fig.add_subplot(2,1,2)
    #
    # ax1.plot(data)
    # ax2.plot(norm_data)
    #
    # plt.show()


## endl
