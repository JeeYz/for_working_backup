import os
import sys

from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.io import wavfile

temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)

from analysis_signal import draw_single_graph, signal_trigger, util_module

train_files_name = 'D:\\train_data_files.txt'
test_files_name = 'D:\\test_data_files.txt'

mod_train_files_name = 'D:\\mod_train_data_files.txt'
mod_test_files_name = 'D:\\mod_test_data_files.txt'

mod_folder_name = 'ASR_train_data_mod'
folder_name = 'ASR_train_data'

## create folder
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

##
def modification_text_file():
    with open(train_files_name, 'r', encoding='utf-8') as fr, \
    open(mod_train_files_name, 'w', encoding='utf-8') as fw:
        while True:
            line = fr.readline()
            if not line: break

            temp = line.split(folder_name)
            new_line = temp[0]+mod_folder_name+temp[1]

            fw.write(new_line)


    with open(test_files_name, 'r', encoding='utf-8') as fr, \
    open(mod_test_files_name, 'w', encoding='utf-8') as fw:
        while True:
            line = fr.readline()
            if not line: break

            temp = line.split(folder_name)
            new_line = temp[0]+mod_folder_name+temp[1]

            fw.write(new_line)

# modification_text_file()


##
def make_data_list():
    full_data_list = list()
    one_file_dict = dict()

    with open(train_files_name, 'r', encoding='utf-8') as f1, \
    open(mod_train_files_name, 'r', encoding='utf-8') as f2:
        while True:
            line = f1.readline()
            line = line.split()
            if not line: break

            line2 = f2.readline()
            line2 = line2.split()

            one_file_dict['file_name'] = line[0]
            one_file_dict['key_label'] = line[1]
            one_file_dict['com_label'] = line[2]
            one_file_dict['mod_file_name'] = line2[0]

            full_data_list.append(one_file_dict)
            one_file_dict = dict()


    with open(test_files_name, 'r', encoding='utf-8') as f1, \
    open(mod_test_files_name, 'r', encoding='utf-8') as f2:
        while True:
            line = f1.readline()
            line = line.split()
            if not line: break

            line2 = f2.readline()
            line2 = line2.split()

            one_file_dict['file_name'] = line[0]
            one_file_dict['key_label'] = line[1]
            one_file_dict['com_label'] = line[2]
            one_file_dict['mod_file_name'] = line2[0]

            full_data_list.append(one_file_dict)
            one_file_dict = dict()

    return full_data_list


# full_data_list = make_data_list()
# print(full_data_list)


##
if __name__ == '__main__':
    aug_train_data_path = 'D:\\aug_train_data.npz'
    aug_test_data_path = 'D:\\aug_test_data.npz'

    train_load_data = np.load(aug_train_data_path)
    train_data = train_load_data['data']
    print(len(train_load_data['data']))
    print(len(train_load_data['label']))

    for one in train_data:
        if len(one) != 199:
            print(len(one))




















## endl
