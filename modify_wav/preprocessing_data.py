import os
import sys
import random

from scipy import signal
from scipy.fft import fft, fftshift
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

train_files_name = 'D:\\train_data_files.txt'
test_files_name = 'D:\\test_data_files.txt'

mod_train_files_name = 'D:\\mod_train_data_files.txt'
mod_test_files_name = 'D:\\mod_test_data_files.txt'

mod_full_data_files_name = 'D:\\mod_full_data_files_list.txt'
mod_full_data_files_name_shuffle = 'D:\\mod_full_data_files_list_shuffle.txt'

mod_train_data_path = 'D:\\mod_train_data.npz'
mod_test_data_path = 'D:\\mod_test_data.npz'



sample_rate = 16000
recording_time = 2
frame_t = 0.025
shift_t = 0.01
buffer_s = 3000


##
def main():
    # full_data_list = make_data_list()
    # wav_to_reduced_data(full_data_list)

    generate_train_data(mod_full_data_files_name)

    load_train_data = np.load(mod_train_data_path, allow_pickle=True)
    # print(load_train_data['label'])
    # print(len(load_train_data['label']))
    load_test_data = np.load(mod_test_data_path, allow_pickle=True)
    # print(load_test_data['label'])
    # print(len(load_test_data['label']))

    # print(load_train_data['data'])

    # for one in load_train_data['data']:
    #     # if len(one) != 199:
    #     #     print(len(one))
    #     print(one)

    temp = load_train_data['data'][1000]
    print(len(temp[0]))
    print(len(temp))
    # print(temp)
    # print(temp[0])
    draw_single_graph.draw_graph_logfbank(temp, sample_rate, title_name='log fb')
    plt.show()

    return


##
def wav_to_reduced_data(data_list):

    for one in data_list:
        one_name = one['file_name']
        mod_one_name = one['mod_file_name']
        samplerate, data = wavfile.read(one_name)
        data = util_module.standardization_func(data)
        mod_data = signal_trigger.evaluate_mean_of_frame(data, frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*recording_time,
                                        threshold_value=0.5)

        mod_directory = mod_one_name.split('\\\\')
        mod_dir_name = '\\\\'.join(mod_directory[:-1])
        createFolder(mod_dir_name)
        # wavfile.write(mod_one_name, samplerate, mod_data)

        if len(mod_data) > sample_rate*recording_time:

            mod_data = signal_trigger.evaluate_mean_of_frame(data, frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*recording_time,
                                        threshold_value=1.0)

            # draw_single_graph.draw_graph_raw_signal(mod_data, title_name='raw signal')

            if len(mod_data) > sample_rate*recording_time:
                print(mod_one_name)
                print(len(mod_data))
                draw_single_graph.draw_graph_raw_signal(mod_data, title_name='raw signal')
                continue

        wavfile.write(mod_one_name, samplerate, mod_data)

    plt.show()

    return


##
def generate_train_data(text_filepath):

    fwb_train = open(mod_train_data_path, 'wb')
    fwb_test = open(mod_test_data_path, 'wb')

    train_data_list = list()
    test_data_list = list()

    train_label_list = list()
    test_label_list = list()

    with open(text_filepath, 'r', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            line = line.split()
            if not line: break

            samplerate, data = wavfile.read(line[0])
            # print(data)
            # print(len(data))
            logfb_feat = logfbank(data)
            # print(len(logfb_feat))
            # print(logfb_feat)

            # transformer = Normalizer().fit(logfb_feat)
            # logfb_feat = transformer.transform(logfb_feat)

            logfb_feat = util_module.standardization_func(logfb_feat)

            temp_path = line[0].split('\\\\')

            if temp_path[2] == 'test':
                test_data_list.append(logfb_feat)
                test_label_list.append(int(line[-1]))
            elif temp_path[2] == 'train':
                one_train = list()
                one_train.append(logfb_feat)
                one_train.append(int(line[-1]))
                train_data_list.append(one_train)

    random.shuffle(train_data_list)

    temp_train_data = list()
    temp_train_label = list()

    for one in train_data_list:
        # print(one)
        temp_train_data.append(one[0])
        temp_train_label.append(one[1])

    train_data = np.asarray(temp_train_data)
    train_label = np.asarray(temp_train_label)

    test_data = np.asarray(test_data_list)
    test_label = np.asarray(test_label_list)

    np.savez_compressed(fwb_train, label=train_label, data=train_data, rate=samplerate)
    np.savez_compressed(fwb_test, label=test_label, data=test_data, rate=samplerate)

    return













## endl
if __name__ == '__main__':
    main()

    # with open(mod_full_data_files_name, 'r', encoding='utf-8') as fr:
    #     line = fr.readline()
    #     line = line.split()
    #
    #     samplerate, data = wavfile.read(line[0])
    #     print(data)
    #     print(len(data))
    #     draw_single_graph.draw_graph_raw_signal(data, title_name='raw data')
    #
    #     logfb_feat = logfbank(data)
    #     print(len(logfb_feat))
    #     print(logfb_feat)
    #     draw_single_graph.draw_graph_logfbank(logfb_feat, sample_rate, title_name='logfb feature')
    #
    #     logfb_feat = util_module.standardization_func(logfb_feat)
    #
    #     draw_single_graph.draw_graph_logfbank(logfb_feat, sample_rate, title_name='stdardized logfb')
    #
    #     plt.show()
