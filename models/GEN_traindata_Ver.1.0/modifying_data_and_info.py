
from global_variables import *


def modifying_start_end_indexs(input_full_data_list):

    for one_file in input_full_data_list:
        for i, one_data in enumerate(one_file['data']):
            temp_start = one_data['start_index']-HEAD_SIZE
            temp_end = one_data['start_index']+HEAD_SIZE

            if temp_start < 0:
                temp_start = 0
            
            if temp_end > FULL_SIZE:
                temp_end = FULL_SIZE
            
            one_data['start_index'] = temp_start
            one_data['end_index'] = temp_end

            one_data['gap_start_end'] = temp_end-temp_start
            

    return input_full_data_list


def fit_full_size_data(input_data):

    gap_size = FULL_SIZE - len(input_data)
    fit_padding = np.zeros(gap_size, dtype=TRAIN_DATA_TYPE)
    result_data = np.append(input_data, fit_padding)

    return result_data





