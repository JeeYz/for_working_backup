
import sys
sys.path.append('D:\\')
temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)
import os

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn import preprocessing
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import random

import librosa
import soundfile

from python_speech_features import logfbank
from files_locations import example_filename
from draw_single_graph import draw_graph_raw_signal, draw_graph_logfbank
from util_module import standardization_func, new_minmax_normal, transpose_the_matrix
from signal_trigger import evaluate_mean_of_frame

from modify_wav import check_files
import analysis_signal_file

mod_full_data_files_name = 'D:\\mod_full_data_files_list.txt'
zeroth_files_list = 'D:\\zeroth_files_list.txt'
zeroth_data_path = 'D:\\voice_data_backup\\zeroth_korean.tar\\zeroth_korean'
zeroth_none_data_path = 'D:\\voice_data_backup\\zeroth_none'

new_none_train_data_zeroth = 'D:\\new_none_train_data_zeroth.txt'

train_data_for_all = 'D:\\train_data_for_all.npz'
train_data_for_all_with_zeroth = 'D:\\train_data_for_all_with_zeroth.npz'


def generate_train_data_for_all_with_zeroth(text_filepath):
    num = 1
    loaded_data = np.load(train_data_for_all)

    fwb_train = open(train_data_for_all_with_zeroth, 'wb')

    train_data_list = list()
    train_label_list = list()

    for one, l in zip(loaded_data['data'], loaded_data['label']):
        one_train = list()
        one_train.append(one)
        one_train.append(l)
        train_data_list.append(one_train)

        print("\rone data complete...\t%d" %num, end='')
        num+=1

    print('\n\n')
    with open(text_filepath, 'r', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            line = line.split()
            if not line: break

            samplerate, data = wavfile.read(line[0])

            if len(data) > 32000:
                print(line[0])
                time.sleep(1000)

            one_train = list()
            logfb_feat = logfbank(data)
            logfb_feat = standardization_func(logfb_feat)
            one_train.append(logfb_feat)
            one_train.append(int(line[-1]))
            train_data_list.append(one_train)

            print("\r one data complete...\t%d" %num, end='')
            num+=1

            if num == 200000:
                break


    random.shuffle(train_data_list)

    temp_train_data = list()
    temp_train_label = list()

    for one in train_data_list:
        # print(one)
        temp_train_data.append(one[0])
        temp_train_label.append(one[1])

    train_data = np.asarray(temp_train_data)
    train_label = np.asarray(temp_train_label)

    np.savez_compressed(fwb_train, label=train_label, data=train_data, rate=samplerate)

    print('\n')
    print(len(train_data))

    return


##
def generate_text_files(target_file_path, data_list):

    with open(target_file_path, 'w', encoding='utf-8') as fw:
        for one in data_list:
            line = '\t\t\t'.join(one)
            fw.write(line+'\n')

    return


##
def generate_zeroth_files_list(data_path, f_format):

    zeroth_path = analysis_signal_file.find_target_files(data_path, f_format)

    new_zeroth_path = list()
    for one in zeroth_path:
        temp = list()
        temp.append(one)
        temp.append(str(0))
        temp.append(str(0))
        temp.append(str(0))
        new_zeroth_path.append(temp)

    return new_zeroth_path


##
def main():
    # data_list = generate_zeroth_files_list(zeroth_none_data_path, '.wav')
    # generate_text_files(new_none_train_data_zeroth, data_list)
    generate_train_data_for_all_with_zeroth(new_none_train_data_zeroth)

    return
















##
if __name__ == '__main__':
    main()

    # loaded_data = np.load(train_data_for_all)
    # print(loaded_data['label'])
    # print(len(loaded_data['label']))
    # print(len(loaded_data['data']))
    #
    # loaded_data = np.load(train_data_for_all_with_zeroth)
    # labels = loaded_data['label']
    # for i in labels:
    #     print(i)







## endl
