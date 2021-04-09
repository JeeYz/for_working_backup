
'''
modifying train data
'''

import os
import sys
sys.path.append('D:\\')
temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)

from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt
import random

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn import preprocessing

from python_speech_features import logfbank
from files_locations import example_filename
from draw_single_graph import draw_graph_logfbank, draw_graph_logfbank_norm, draw_graph_raw_signal
import draw_single_graph
from util_module import standardization_func, new_minmax_normal, transpose_the_matrix

mod_full_data_files_name = 'D:\\mod_full_data_files_list.txt'
train_data_none_devided_2 = 'D:\\train_data_none_devided_2.npz'
train_data_for_all = 'D:\\train_data_for_all.npz'
train_data_all_none_devided_2 = 'D:\\train_data_all_none_devided_2.npz'

zeroth_path_devided_2 = 'D:\\voice_data_backup\\zeroth_none_devided_2'


## command
# new_ver_train_data_path = 'D:\\new_ver_train_data\\command_train_data.npz'
# label_mapping_dict = {4:[1, 'call'],
#                     5:[2, 'camera'],
#                     10:[3, 'picture'],
#                     13:[4, 'record'],
#                     15:[5, 'stop'],
#                     7:[6, 'end'],
#                     0:[0, 'None']}
# ## call
# new_ver_train_data_path = 'D:\\new_ver_train_data\\call_command_train_data.npz'
# label_mapping_dict = {4:[1, 'call'],
#                     0:[0, 'None']}
#
# ## camera
# new_ver_train_data_path = 'D:\\new_ver_train_data\\camera_command_train_data.npz'
# label_mapping_dict = {5:[1, 'camera'],
#                     0:[0, 'None']}
#
# ## picture
# new_ver_train_data_path = 'D:\\new_ver_train_data\\picture_command_train_data.npz'
# label_mapping_dict = {10:[1, 'picture'],
#                     0:[0, 'None']}
#
# ## record
# new_ver_train_data_path = 'D:\\new_ver_train_data\\record_command_train_data.npz'
# label_mapping_dict = {13:[1, 'record'],
#                     0:[0, 'None']}
#
# ## stop
# new_ver_train_data_path = 'D:\\new_ver_train_data\\stop_command_train_data.npz'
# label_mapping_dict = {15:[1, 'stop'],
#                     0:[0, 'None']}
#
# ## end
# new_ver_train_data_path = 'D:\\new_ver_train_data\\end_command_train_data.npz'
# label_mapping_dict = {7:[1, 'end'],
#                     0:[0, 'None']}
#
## keyword
# new_ver_train_data_path = 'D:\\new_ver_train_data\\keyword_train_data.npz'
# label_mapping_dict = {16:[1, 'hipnc'],
#                     0:[0, 'None']}


##
def find_target_files(fpath, file_format):

    files_list = list()

    for (path, dir, files) in os.walk(fpath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_format:
                files_list.append(path+'\\'+filename)

    return files_list


##
def change_label_num(ori_num):

    return label_mapping_dict[ori_num][0]


##
def make_integrate_all_data_dif_ver():

    fwb_train = open(new_ver_train_data_path, 'wb')

    train_loaded_data = np.load(train_data_for_all)
    # print(len(train_loaded_data['data']))
    train_loaded_data_data = train_loaded_data['data']
    train_loaded_data_label = train_loaded_data['label']

    # none_loaded_data = np.load(train_data_none_devided_2)
    # # print(len(none_loaded_data['data']))
    # none_loaded_data_data = none_loaded_data['data']
    # none_loaded_data_label = none_loaded_data['label']

    train_list = list()

    num = 1
    for data, label in zip(train_loaded_data_data, train_loaded_data_label):
        train_list.append([data, label])
        print('\rone data was processed...\t%d'%num, end='')
        num+=1

    print('\ndone...')

    # for data, label in zip(none_loaded_data_data, none_loaded_data_label):
    #     train_list.append([data, label])
    #     print('\rone data was processed...\t%d'%num, end='')
    #     num+=1
    #     if num == 200000:
    #         break

    # print('\ndone...')

    random.shuffle(train_list)

    temp_train_data = list()
    temp_train_label = list()

    num = 1
    for one in train_list:
        # print(one)
        temp_train_data.append(one[0])
        if one[1] in label_mapping_dict.keys():
            temp_train_label.append(change_label_num(one[1]))
        else:
            temp_train_label.append(0)

        print('\rone data was processed...\t%d'%num, end='')
        num+=1

    print('\ndone...')

    train_data = np.asarray(temp_train_data)
    train_label = np.asarray(temp_train_label)

    np.savez_compressed(fwb_train, label=train_label, data=train_data, rate=16000)

    print(len(train_data))

    return


##
def make_integrate_all_data():

    fwb_train = open(train_data_all_none_devided_2, 'wb')

    train_loaded_data = np.load(train_data_for_all)
    # print(len(train_loaded_data['data']))
    train_loaded_data_data = train_loaded_data['data']
    train_loaded_data_label = train_loaded_data['label']

    none_loaded_data = np.load(train_data_none_devided_2)
    # print(len(none_loaded_data['data']))
    none_loaded_data_data = none_loaded_data['data']
    none_loaded_data_label = none_loaded_data['label']

    train_list = list()

    for data, label in zip(train_loaded_data_data, train_loaded_data_label):
        train_list.append([data, label])

    print('done...')

    for data, label in zip(none_loaded_data_data, none_loaded_data_label):
        train_list.append([data, label])

    print('done...')

    random.shuffle(train_list)

    temp_train_data = list()
    temp_train_label = list()

    for one in train_list:
        # print(one)
        temp_train_data.append(one[0])
        temp_train_label.append(one[1])

    print('done...')

    train_data = np.asarray(temp_train_data)
    train_label = np.asarray(temp_train_label)

    np.savez_compressed(fwb_train, label=train_label, data=train_data, rate=16000)

    print(len(train_data))

    return


## main
def main():
    num = 1
    with open(mod_full_data_files_name, 'r', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            line = line.split()
            if not line: break

            if num == 3000:
                sr, data = wavfile.read(line[0])
                draw_graph_raw_signal(data, title_name='raw')
                plt.show()

            num+=1

    return


##














## main
if __name__ == '__main__':
    # main()

    # ## none label devided 2 ver.
    # fwb_train = open(train_data_none_devided_2, 'wb')
    # ##
    # zeroth_devided_2 = find_target_files(zeroth_path_devided_2, '.wav')
    #
    # data_list = list()
    # label_list = list()
    #
    # for i,one in enumerate(zeroth_devided_2):
    #     sr, data = wavfile.read(one)
    #     logfb_feat = logfbank(data)
    #     logfb_feat = standardization_func(logfb_feat)
    #
    #     if len(logfb_feat) != 199:
    #         print(len(logfb_feat))
    #         time.sleep(1000)
    #
    #     data_list.append(logfb_feat)
    #     label_list.append(0)
    #     print('\rone file is done...\t%d'%(i+1), end='')
    #
    # ##
    # np.savez_compressed(fwb_train, label=label_list, data=data_list, rate=16000)


    make_integrate_all_data_dif_ver()


## endl
