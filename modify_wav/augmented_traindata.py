
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

import librosa

mod_full_data_files_name = 'D:\\mod_full_data_files_list.txt'

aug_train_data_path = 'D:\\aug_train_data.npz'
aug_test_data_path = 'D:\\aug_test_data.npz'

train_data_for_all = 'D:\\train_data_for_all.npz'

train_data_sampling_one = 'D:\\train_data_sampling_one.npz'
test_data_sampling_one = 'D:\\test_data_sampling_one.npz'

sample_rate = 16000
recording_time = 2
frame_t = 0.025
shift_t = 0.01
buffer_s = 3000

rate_list = [
0.97, 0.94, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70, 0.67, 0.64, 0.61,
1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33, 1.36, 1.39,
1.0]

def main():
    generate_train_data_for_all(mod_full_data_files_name)

    return

#%%
def time_stretch(data, aug_rate):

    aug_data = librosa.effects.time_stretch(data, aug_rate)
    print(aug_data)

    return aug_data


## simply augmented
## train and test
def generate_train_data(text_filepath):

    fwb_train = open(aug_train_data_path, 'wb')
    fwb_test = open(aug_test_data_path, 'wb')

    train_data_list = list()
    test_data_list = list()

    train_label_list = list()
    test_label_list = list()

    with open(text_filepath, 'r', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            line = line.split()
            if not line: break

            temp_path = line[0].split('\\\\')

            if temp_path[2] == 'test':
                samplerate, data = wavfile.read(line[0])
                logfb_feat = logfbank(data)
                logfb_feat = util_module.standardization_func(logfb_feat)
                test_data_list.append(logfb_feat)
                test_label_list.append(int(line[-1]))

            elif temp_path[2] == 'train':
                samplerate, data = wavfile.read(line[0])
                for r in rate_list:
                    one_train = list()
                    if r == 1.0:
                        logfb_feat = logfbank(data)
                        logfb_feat = util_module.standardization_func(logfb_feat)
                        one_train.append(logfb_feat)
                        one_train.append(int(line[-1]))
                        train_data_list.append(one_train)
                    else:
                        # aug_data = time_stretch(data, r)
                        aug_data = librosa.effects.time_stretch(data, r)
                        if len(aug_data) < sample_rate*recording_time:
                            aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                            frame_time=frame_t,
                                            shift_time=shift_t,
                                            sample_rate=sample_rate,
                                            buffer_size=buffer_s,
                                            full_size = sample_rate*recording_time,
                                            threshold_value=0.5)

                        elif len(aug_data) > sample_rate*recording_time:
                            aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                            frame_time=frame_t,
                                            shift_time=shift_t,
                                            sample_rate=sample_rate,
                                            buffer_size=buffer_s,
                                            full_size = sample_rate*recording_time,
                                            threshold_value=1.0)
                        if len(aug_data) > sample_rate*recording_time:
                            continue

                        logfb_feat = logfbank(aug_data)
                        logfb_feat = util_module.standardization_func(logfb_feat)
                        one_train.append(logfb_feat)
                        one_train.append(int(line[-1]))
                        train_data_list.append(one_train)

            print('one data complete...')

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

    print(len(train_data))

    return


## augmented all in one (train + test)
def generate_train_data_for_all(text_filepath):

    fwb_train = open(train_data_for_all, 'wb')

    train_data_list = list()
    train_label_list = list()

    num = 1
    with open(text_filepath, 'r', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            line = line.split()
            if not line: break

            samplerate, data = wavfile.read(line[0])
            for r in rate_list:
                one_train = list()
                if r == 1.0:
                    logfb_feat = logfbank(data)
                    logfb_feat = util_module.standardization_func(logfb_feat)
                    one_train.append(logfb_feat)
                    one_train.append(int(line[-1]))
                    train_data_list.append(one_train)
                else:
                    # aug_data = time_stretch(data, r)
                    aug_data = librosa.effects.time_stretch(data, r)
                    if len(aug_data) < sample_rate*recording_time:
                        aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                        frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*recording_time,
                                        threshold_value=0.5)

                    elif len(aug_data) > sample_rate*recording_time:
                        aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                        frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*recording_time,
                                        threshold_value=1.0)
                    if len(aug_data) > sample_rate*recording_time:
                        continue

                    logfb_feat = logfbank(aug_data)
                    logfb_feat = util_module.standardization_func(logfb_feat)
                    one_train.append(logfb_feat)
                    one_train.append(int(line[-1]))
                    train_data_list.append(one_train)

            print('\rone data complete... %d'% num, end='')
            num+=1

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

    print(len(train_data))

    return


## augmented 90% in one (train + test)
def generate_train_data_for_all_1(text_filepath):

    fwb_train = open(train_data_sampling_one, 'wb')
    fwb_test = open(test_data_sampling_one, 'wb')

    train_data_list = list()
    test_data_list = list()

    train_label_list = list()
    test_label_list = list()

    num = 1
    with open(text_filepath, 'r', encoding='utf-8') as fr:
        comm = 'first_c'
        person = 'first_p'

        while True:
            line = fr.readline()
            line = line.split()
            if not line: break

            temp_path = line[0].split('\\\\')

            if temp_path[-2] != comm or temp_path[-3] != person:
                samplerate, data = wavfile.read(line[0])
                logfb_feat = logfbank(data)
                logfb_feat = util_module.standardization_func(logfb_feat)
                test_data_list.append(logfb_feat)
                test_label_list.append(int(line[-1]))
                comm = temp_path[-2]
                person = temp_path[-3]


            else:
                samplerate, data = wavfile.read(line[0])
                for r in rate_list:
                    one_train = list()
                    if r == 1.0:
                        logfb_feat = logfbank(data)
                        logfb_feat = util_module.standardization_func(logfb_feat)
                        one_train.append(logfb_feat)
                        one_train.append(int(line[-1]))
                        train_data_list.append(one_train)
                    else:
                        # aug_data = time_stretch(data, r)
                        aug_data = librosa.effects.time_stretch(data, r)
                        if len(aug_data) < sample_rate*recording_time:
                            aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                            frame_time=frame_t,
                                            shift_time=shift_t,
                                            sample_rate=sample_rate,
                                            buffer_size=buffer_s,
                                            full_size = sample_rate*recording_time,
                                            threshold_value=0.5)

                        elif len(aug_data) > sample_rate*recording_time:
                            aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                            frame_time=frame_t,
                                            shift_time=shift_t,
                                            sample_rate=sample_rate,
                                            buffer_size=buffer_s,
                                            full_size = sample_rate*recording_time,
                                            threshold_value=1.0)
                        if len(aug_data) > sample_rate*recording_time:
                            continue

                        logfb_feat = logfbank(aug_data)
                        logfb_feat = util_module.standardization_func(logfb_feat)
                        one_train.append(logfb_feat)
                        one_train.append(int(line[-1]))
                        train_data_list.append(one_train)

            print('one data complete...', num)
            num+=1

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

    print(len(train_data))

    return


def return_random_num():
    rate = random.choice([1.03, 1.02, 1.01, 1.0, 0.99, 0.98, 0.97])
    rand_size = random.randrange(200, 240)
    return rate, rand_size


##
def cut_raw_data(data, **kwargs):

    if "frame_time" in kwargs.keys():
        frame_time = kwargs['frame_time']
    if "shift_time" in kwargs.keys():
        shift_time = kwargs['shift_time']
    if "sample_rate" in kwargs.keys():
        sr = kwargs['sample_rate']
    if "buffer_size" in kwargs.keys():
        buf_size = kwargs['buffer_size']
    if "full_size" in kwargs.keys():
        full_size = kwargs['full_size']
    if "threshold_value" in kwargs.keys():
        trigger_val = kwargs['threshold_value']

    frame_size = int(sr*frame_time)
    shift_size = int(sr*shift_time)

    num_frames = len(data)//(frame_size-shift_size)+1

    mean_val_list = list()

    for i in range(num_frames):
        temp_n = i*(frame_size-shift_size)
        if temp_n+frame_size > len(data):
            one_frame_data = data[temp_n:len(data)]
        else:
            one_frame_data = data[temp_n:temp_n+frame_size]
        mean_val_list.append(np.mean(np.abs(one_frame_data)))

    for i,start in enumerate(mean_val_list):
        if trigger_val < start:
            start_index = i
            break
        else:
            start_index = 0

    for i,end in enumerate(reversed(mean_val_list)):
        if trigger_val < end:
            end_index = len(mean_val_list)-i
            break
        else:
            end_index = len(mean_val_list)

    ##
    temp_signal_data = list()


    temp = (frame_size-shift_size)*start_index-buf_size
    if temp <= 0:
        temp = 0

    result = data[temp:(frame_size-shift_size)*end_index+buf_size]

    if full_size > len(result):
        result = fit_determined_size(result, full_size=full_size)
        result = add_noise_data(result, full_size=full_size)

    return result


## augment all
## time stretch random frame
def generate_train_data_for_all_2(text_filepath):

    fwb_train = open(train_data_for_all, 'wb')

    train_data_list = list()
    train_label_list = list()

    num = 1
    with open(text_filepath, 'r', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            line = line.split()
            if not line: break

            samplerate, data = wavfile.read(line[0])



            for r in rate_list:
                one_train = list()
                if r == 1.0:
                    logfb_feat = logfbank(data)
                    logfb_feat = util_module.standardization_func(logfb_feat)
                    one_train.append(logfb_feat)
                    one_train.append(int(line[-1]))
                    train_data_list.append(one_train)
                else:
                    # aug_data = time_stretch(data, r)
                    aug_data = librosa.effects.time_stretch(data, r)
                    if len(aug_data) < sample_rate*recording_time:
                        aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                        frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*recording_time,
                                        threshold_value=0.5)

                    elif len(aug_data) > sample_rate*recording_time:
                        aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                        frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*recording_time,
                                        threshold_value=1.0)
                    if len(aug_data) > sample_rate*recording_time:
                        continue

                    logfb_feat = logfbank(aug_data)
                    logfb_feat = util_module.standardization_func(logfb_feat)
                    one_train.append(logfb_feat)
                    one_train.append(int(line[-1]))
                    train_data_list.append(one_train)

            print('one data complete...', num)
            num+=1

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

    print(len(train_data))

    return










## endl
if __name__ == '__main__':
    # main()

    loaded_data = np.load(train_data_for_all)

    num = 0
    for one in loaded_data['data']:
        if len(one) != 199:
            num += 1
            print("\r%d data is corrupted" % num , end='')
            print(len(one))
            time.sleep(1000)

    print("\ndone...")







## endl
