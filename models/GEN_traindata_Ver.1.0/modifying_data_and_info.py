
from global_variables import *


def modifying_start_end_indexs(input_data_list):
    for one_file in input_data_list :
        for i, one_data in enumerate(one_file['file_data']):
            temp_start = one_data['start_index']-HEAD_SIZE
            temp_end = one_data['end_index']+TAIL_SIZE

            if temp_start < 0:
                temp_start = 0
            
            if temp_end > FULL_SIZE:
                temp_end = FULL_SIZE
            
            one_data['start_index'] = temp_start
            one_data['end_index'] = temp_end

            one_data['gap_start_end'] = temp_end-temp_start


def fit_full_size_data(input_data):

    gap_size = FULL_SIZE - len(input_data)
    fit_padding = np.zeros(gap_size, dtype=TRAIN_DATA_TYPE)
    result_data = np.append(input_data, fit_padding)

    return result_data


def make_fit_full_size_data_mid(input_data_list):
    for one_file in input_data_list:
        for one_data in one_file['file_data']:
            init_data = copy.deepcopy(one_data['data'])
            init_start = one_data['start_index']
            init_end = one_data['end_index']
            mid_val = (init_start+init_end)//2

            half_full_size = FULL_SIZE//2

            head_zeros = np.zeros(half_full_size, dtype=TRAIN_DATA_TYPE)

            new_data = np.append(head_zeros, init_data)
            new_data = np.append(new_data, head_zeros)

            new_mid = mid_val+half_full_size

            new_start = new_mid-half_full_size
            new_end = new_mid+half_full_size

            new_data = new_data[new_start:new_end]

            one_data['data'] = new_data
            one_data['data_length'] = len(new_data)
            one_data['start_index'] = None
            one_data['end_index'] = None
            one_data['gap_start_end'] = None


def make_fit_full_size_data(input_data_list):
    for one_file in input_data_list:
        for one_data in one_file['file_data']:
            init_data = copy.deepcopy(one_data['data'])
            init_start = one_data['start_index']
            init_end = one_data['end_index']

            half_full_size = FULL_SIZE//2
            tail_zeros = np.zeros(half_full_size, dtype=TRAIN_DATA_TYPE)

            new_data = np.append(init_data, tail_zeros)

            new_start = init_start-BUFFER_SIZE

            if new_start < 0:
                new_start=0
            
            new_end = new_start+FULL_SIZE

            new_data = new_data[new_start:new_end]

            one_data['data'] = new_data
            one_data['data_length'] = len(new_data)
            one_data['start_index'] = None
            one_data['end_index'] = None
            one_data['gap_start_end'] = None
            


