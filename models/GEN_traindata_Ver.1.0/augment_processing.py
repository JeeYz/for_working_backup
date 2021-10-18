
from scipy.ndimage.interpolation import shift
from global_variables import *


class Augment_Process():
    def __init__(self):
        pass


    def time_stretch_process(self, files_list):

        for one_file in files_list:
            data = copy.deepcopy(one_file['data'][0])
            for rate in rate_list:
                aug_data = librosa.effects.time_stretch(data, rate)
                one_file['data'].append(aug_data)

        return files_list


    def make_one_data_dictionary(self, ):
        return 


    def add_zero_padding_back(self, input_data):
        temp_tail_num = FULL_SIZE - len(input_data)
        zero_padding = np.zeros(temp_tail_num, dtype=TRAIN_DATA_TYPE)
        result_data = np.append(input_data, zero_padding)
        return result_data


    def random_position_process(self, input_files_list):

        for one_file in input_files_list:
            auged_data_list = list()
            for one_data in one_file['data']:

                init_start = one_data['start_index']
                init_end = one_data['end_index']
                init_length = one_data['data_length']
                init_data = one_data['data']

                for i in range(DATA_AUG_POSITION):
                    shift_value_of_start = i*BLOCK_OF_RANDOM
                    temp_start = init_start-shift_value_of_start

                    temp_dict = dict()

                    if temp_start < 0:
                        break
                    
                    temp_end_of_data = temp_start+FULL_SIZE

                    cond = len(init_data) < temp_end_of_data
                    if cond:
                        temp_data = self.add_zero_padding_back(init_data[temp_start:])
                    else:
                        temp = temp_start+FULL_SIZE
                        temp_data = init_data[temp_start:temp]

                    if len(temp_data) != FULL_SIZE:
                        raise Exception("데이터의 길이가 정해진 길이가 아닙니다.")

                    temp_dict['data'] = temp_data
                    temp_dict['auged_condition'] = shift_value_of_start

                    auged_data_list.append(temp_dict)                    

            one_file['data'] = auged_data_list

        return input_files_list


    def add_zero_padding_front_end(self, input_data):

        return result_data


    def random_position_process_zero_padding(self, input_files_list):

        for one_file in input_files_list:
            auged_data_list = list()
            for one_data in one_file['data']:

                init_start = one_data['start_index']
                init_end = one_data['end_index']
                init_length = one_data['data_length']
                init_data = one_data['data']

                for i in range(DATA_AUG_POSITION):
                    shift_value_of_start = i*BLOCK_OF_RANDOM
                    temp_start = init_start-shift_value_of_start
                    temp_end = temp_start+init_length

                    temp_dict = dict()

                    if temp_start < 0:
                        break

                    mid_data = init_data[temp_start:temp_end]

                    # tomorrow job

                    if len(temp_data) != FULL_SIZE:
                        raise Exception("데이터의 길이가 정해진 길이가 아닙니다.")

                    temp_dict['data'] = temp_data
                    temp_dict['auged_condition'] = shift_value_of_start

                    auged_data_list.append(temp_dict)                    

            one_file['data'] = auged_data_list

        return input_files_list



