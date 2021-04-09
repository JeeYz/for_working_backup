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
import time

from python_speech_features import logfbank
from files_locations import example_filename
from draw_single_graph import draw_graph_raw_signal, draw_graph_logfbank
from util_module import standardization_func, new_minmax_normal, transpose_the_matrix
from signal_trigger import evaluate_mean_of_frame

from modify_wav import check_files

mod_full_data_files_name = 'D:\\mod_full_data_files_list.txt'
zeroth_files_list = 'D:\\zeroth_files_list.txt'
zeroth_data_path = 'D:\\voice_data_backup\\zeroth_korean.tar\\zeroth_korean'
zeroth_none_data_path = 'D:\\voice_data_backup\\zeroth_none'
zeroth_none_devided_path_2 = 'D:\\voice_data_backup\\zeroth_none_devided_2'
full_new_train_data_with_none = 'D:\\full_new_train_data_with_none.txt'

devided_none_data_2 = 'D:\\divided_none_data_2.npz'

sample_rate = 16000
threshold = 0.5
frame_t = 0.025
shift_t = 0.01
buffer_s = 3000
recording_time = 2


#%%
def find_target_files(fpath, file_format):

    files_list = list()

    for (path, dir, files) in os.walk(fpath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_format:
                files_list.append(path+'\\'+filename)

    return files_list


##
def write_wav_files(files_list):

    for one in files_list:
        data, sr = soundfile.read(one)
        if sr != 16000:
            print(sr, one)
        else:
            temp_path = one.split('.')
            temp_path[-1] = 'wav'
            temp_path = '.'.join(temp_path)
            # print(temp_path)
            soundfile.write(temp_path, data, sr, format='wav', subtype='PCM_16')

    return


##
def add_buffer(data, len_data):

    noise_data = np.random.randn(buffer_s-len_data)*0.01
    result = list()
    result.extend(noise_data)
    result.extend(data)

    return result


## devide determined size(random 10000~20000)
def devide_data_2(data):

    data_list = list()

    threshold = 0.5
    min_len = 10
    max_len = 20
    buf_size = 3000

    frame_size = int(sample_rate*0.025)
    shift_size = int(sample_rate*0.01)

    num_frames = len(data)//(frame_size-shift_size)+1

    data = standardization_func(data)

    # draw_graph_raw_signal(data, title_name='raw')
    # plt.show()
    #
    # time.sleep(1000)

    temp = list()
    temp_list = list()
    low_temp = list()

    mean_val_list = list()

    for i in range(num_frames):
        temp_n = i*(frame_size-shift_size)
        if temp_n+frame_size > len(data):
            one_frame_data = data[temp_n:len(data)]
        else:
            one_frame_data = data[temp_n:temp_n+frame_size]
        mean_val_list.append(np.mean(np.abs(one_frame_data)))

    for i,start in enumerate(mean_val_list):
        if threshold < start:
            start_index = i
            break
        else:
            start_index = 0

    for i,end in enumerate(reversed(mean_val_list)):
        if threshold < end:
            end_index = len(mean_val_list)-i
            break
        else:
            end_index = len(mean_val_list)

    temp = (frame_size-shift_size)*start_index-buf_size
    if temp <= 0:
        temp = 0

    result = data[temp:(frame_size-shift_size)*end_index+buf_size]

    # draw_graph_raw_signal(result, title_name='raw')
    # plt.show()
    # time.sleep(1000)

    # random.randrange(min_len, max_len)*1000

    start = 0
    end = 0
    data_list = list()

    while True:
        temp = random.randrange(min_len, max_len)*1000
        end = start+temp

        if end > len(result):
            end = len(result)

        if min_len*1000 > end-start:
            break

        temp_data = result[start:end]
        noise_data = np.random.randn(buf_size)*0.01
        # noise_data = noise_data.tolist()
        # temp_data = noise_data.extend(temp_data)
        temp_data = np.append(noise_data, temp_data, axis=0)
        # temp_data = evaluate_mean_of_frame(temp_data,
        #                 frame_time=0.025,
        #                 shift_time=0.01,
        #                 sample_rate=16000,
        #                 buffer_size=3000,
        #                 full_size=32000,
        #                 threshold_value=0.5)
        # one_data = list()

        # logfb_feat = logfbank(temp_data)
        # logfb_feat = standardization_func(logfb_feat)

        # one_data.append(logfb_feat)
        # one_data.append(temp_data)
        # one_data.append(0)
        data_list.append(temp_data)

        # draw_graph_raw_signal(temp_data, title_name='raw')
        # draw_graph_logfbank(logfb_feat, 16000, title_name='logfb')

        start = end



    # draw_graph_raw_signal(result, title_name='raw')
    # plt.show()
    # print(len(result))
    # time.sleep(1000)

    return data_list


## devide algorithm #2
def devide_data_1(data):

    data_list = list()

    add_num = 5
    min_len = 10
    max_len = 20

    frame_size = int(sample_rate*0.025)
    shift_size = int(sample_rate*0.01)

    num_frames = len(data)//(frame_size-shift_size)+1

    data = standardization_func(data)

    temp = list()
    temp_list = list()
    low_temp = list()

    for i in range(num_frames):
        temp_n = i*(frame_size-shift_size)
        if temp_n+frame_size > len(data):
            one_frame_data = data[temp_n:len(data)]
        else:
            one_frame_data = data[temp_n:temp_n+frame_size-shift_size]

            if threshold < np.mean(np.abs(one_frame_data)):
                if low_temp != [] and len(low_temp) < add_num*(frame_size-shift_size):
                    temp.extend(low_temp)

                temp.extend(one_frame_data)
                low_temp = list()
            else:
                if temp != [] and len(low_temp) > add_num*(frame_size-shift_size):
                    st_num = temp_n-len(temp)-(add_num-1)*(frame_size-shift_size)
                    end_num = temp_n+frame_size-shift_size-len(temp)
                    front_data = data[st_num:end_num]

                    new_list = list()
                    new_list.extend(front_data)
                    new_list.extend(temp)
                    temp = add_buffer(new_list, add_num*(frame_size-shift_size))
                    temp.extend(low_temp)
                    data_list.append(temp)
                    temp = list()
                    # print(len(temp))

                low_temp.extend(one_frame_data)
                # print(len(low_temp))


    # print(len(data_list))

    new_data_list = list()

    for one in data_list:
        if len(one) > at_least_len and len(one) < max_len:
            new_data_list.append(one)

    # print(data_list)
    # for one in new_data_list:
    #     draw_graph_raw_signal(one, title_name='raw')
    #     print(len(one))
    #
    # draw_graph_raw_signal(data, title_name='raw')
    #
    # plt.show()
    # time.sleep(10000)

    return new_data_list


##
def devide_data(data):

    data_list = list()

    add_num = 5
    at_least_len = 7000
    max_len = 20000

    frame_size = int(sample_rate*0.025)
    shift_size = int(sample_rate*0.01)

    num_frames = len(data)//(frame_size-shift_size)+1

    data = standardization_func(data)

    temp = list()
    temp_list = list()
    low_temp = list()

    for i in range(num_frames):
        temp_n = i*(frame_size-shift_size)
        if temp_n+frame_size > len(data):
            one_frame_data = data[temp_n:len(data)]
        else:
            one_frame_data = data[temp_n:temp_n+frame_size-shift_size]

            if threshold < np.mean(np.abs(one_frame_data)):
                if low_temp != [] and len(low_temp) < add_num*(frame_size-shift_size):
                    temp.extend(low_temp)

                temp.extend(one_frame_data)
                low_temp = list()
            else:
                if temp != [] and len(low_temp) > add_num*(frame_size-shift_size):
                    st_num = temp_n-len(temp)-(add_num-1)*(frame_size-shift_size)
                    end_num = temp_n+frame_size-shift_size-len(temp)
                    front_data = data[st_num:end_num]

                    new_list = list()
                    new_list.extend(front_data)
                    new_list.extend(temp)
                    temp = add_buffer(new_list, add_num*(frame_size-shift_size))
                    temp.extend(low_temp)
                    data_list.append(temp)
                    temp = list()
                    # print(len(temp))

                low_temp.extend(one_frame_data)
                # print(len(low_temp))


    # print(len(data_list))

    new_data_list = list()

    for one in data_list:
        if len(one) > at_least_len and len(one) < max_len:
            new_data_list.append(one)

    # print(data_list)
    # for one in new_data_list:
    #     draw_graph_raw_signal(one, title_name='raw')
    #     print(len(one))
    #
    # draw_graph_raw_signal(data, title_name='raw')
    #
    # plt.show()
    # time.sleep(10000)

    return new_data_list


##
def refine_data(data_list):
    result_list = list()
    for one in data_list:
        refine_d = evaluate_mean_of_frame(one,
                        frame_time=frame_t,
                        shift_time=shift_t,
                        sample_rate=sample_rate,
                        buffer_size=buffer_s,
                        full_size = sample_rate*recording_time,
                        threshold_value=threshold)

        result_list.append(refine_d)
        if len(refine_d) != 32000:
            print(one)
            print(len(refine_d))
            time.sleep(1000)


    #     draw_graph_raw_signal(refine_d, title_name='raw')
    #
    # plt.show()
    # time.sleep(10000)

    return result_list


##
def generate_wav_files(data_list, target_path):
    sr = 16000
    for i,one in enumerate(data_list):
        num = str(i)
        soundfile.write(target_path+'_'+num+'.wav', one, sr, format='wav', subtype='PCM_16')

    return


##
def make_none_data(file_path, target_path):
    sr, data = wavfile.read(file_path)
    # print(data)
    # print(file_path, sr)
    # data = standardization_func(data)
    # draw_graph_raw_signal(data, title_name='raw')
    # plt.show()
    # time.sleep(1000)

    result = devide_data_2(data)
    result = refine_data(result)
    generate_wav_files(result, target_path)

    return


##
def main():

    # flac_list = find_target_files(zeroth_data_path, '.flac')
    # write_wav_files(flac_list)

    zeroth_list = find_target_files(zeroth_data_path, '.wav')

    for i,one in enumerate(zeroth_list):
        # if i == 3000:
        #     break
        temp_path = one.split('zeroth_korean')
        temp_fn = temp_path[-1].split('\\')[-1].split('.')[-2]
        temp = temp_path[-1].split('\\')[:-1]
        temp = '\\'.join(temp)

        folder_path = zeroth_none_devided_path_2 + '\\' + temp
        check_files.createFolder(folder_path)

        file_path = folder_path + '\\' + temp_fn
        make_none_data(one, file_path)
        print('\rone file is done....\t%d' %i, end='')

    # draw_graph_raw_signal(data, title_name='raw')
    # plt.show()

    return
















##
if __name__ == '__main__':
    main()
