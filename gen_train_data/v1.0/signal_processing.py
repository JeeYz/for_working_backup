
from global_variables import *
import trigger_algorithm as trig


def read_each_files_to_data(files_list):

    for one_file in files_list:
        filename = one_file['filename']
        sr, data = wavfile.read(filename)
        data = np.array(data, dtype=TRAIN_DATA_TYPE)

        one_file['data'] = list()
        one_file['data'].append(data)

    return files_list


def detect_saturation_signal(files_list):

    deleted_num= 0

    length_of_list = len(files_list)
    print("start detecting saturation signals...")
    print("input data length : {length}".format(length=length_of_list ))

    for one_file in reversed(files_list):
        filename = one_file['filename']
        sr, data = wavfile.read(filename)

        max_value = np.max(data)
        min_value = np.min(data)

        cond1 = max_value < MAX_SIGNAL_VALUE
        cond2 = min_value > MIN_SIGNAL_VALUE
        if cond1 and cond2:
            pass
        else:
            idx_num = files_list.index(one_file)
            del files_list[idx_num]
            deleted_num  += 1
        
    length_of_list  = len(files_list)
    print("output data length : {length}".format(length=length_of_list ))
    print("{dnum} data is deleted...".format(dnum=deleted_num))
    print('\n')

    return files_list


def standardize_data(files_list):
    for one_data in files_list:
        if one_data['data'] is not []:
            for i, one_aug in enumerate(one_data['data']):
                tmp_data = copy.deepcopy(one_aug)
                one_data['data'][i] = (tmp_data-np.mean(tmp_data))/np.std(tmp_data)

    return files_list


def normalize_data(files_list):
    for one_data in files_list:
        if one_data['data'] is not []:
            for i, one_aug in enumerate(one_data['data']):
                tmp_data = copy.deepcopy(one_aug)
                min_val = np.min(tmp_data)
                max_val = np.max(tmp_data)
                one_data ['data'][i] = ((tmp_data-min_val)/(max_val-min_val)-0.5)*2

    return files_list


def simple_norm_data(files_list):

    for one_data in files_list:
        if one_data['data'] is not []:
            for i, one_aug in enumerate(one_data['data']):
                tmp_data = copy.deepcopy(one_aug)
                one_data ['data'][i] = np.array(tmp_data, dtype=TRAIN_DATA_TYPE)/MAX_SIGNAL_VALUE

    return files_list










