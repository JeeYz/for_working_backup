
from global_variables import *


def none_aug_process(input_files_list):
    half_size = FULL_SIZE//2
    
    for one_file in input_files_list:
        data_list = list()
        
        for one_data in one_file['data']:
            if one_data['gap_start_end'] > FULL_SIZE:
                continue

            temp_dict = dict()

            init_data = one_data['data']
            init_start = one_data['start_index']
            init_end = one_data['end_index']
            init_gap_size = one_data['gap_start_end']

            mid_position = (init_start+init_end)//2+half_size

            zeros_front = np.zeros(half_size, dtype=TRAIN_DATA_TYPE)
            zeros_tail = np.zeros(half_size, dtype=TRAIN_DATA_TYPE)

            temp_data = np.append(zeros_front, init_data)
            temp_data = np.append(temp_data, zeros_tail)

            new_start = mid_position-half_size            
            new_end = mid_position+half_size

            temp_data = temp_data[new_start:new_end]

            temp_dict['data'] = temp_data
            temp_dict['position_value'] = -1
            temp_dict['data_length'] = len(temp_data)
            temp_dict['label'] = one_file['label']

            data_list.append(temp_dict)
        
        if data_list is []:
            continue

        one_file['data'] = data_list

    return input_files_list

