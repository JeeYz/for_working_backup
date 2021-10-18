
from global_variables import *


def modifying_start_end_indexs(input_full_data_list):

    for one_file in input_full_data_list:
        for i, one_data in enumerate(one_file['data']):
            one_data['start_index'] -= HEAD_SIZE
            one_data['end_index'] += TAIL_SIZE
            one_data['gap_start_end'] += (HEAD_SIZE+TAIL_SIZE)

    return input_full_data_list





