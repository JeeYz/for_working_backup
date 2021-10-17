
from numpy.core.fromnumeric import mean
from global_variables import *


class Signal_Trigger():
    def __init__(self):
        pass


    def apply_trigger_algorithm(self, result_list):

        for one_file in result_list:
            for i,one_data in enumerate(one_file['data']):
                result_dict = self.signal_trigger_algorithm(one_data)
                one_file['data'][i] = result_dict

        return result_list


    def signal_trigger_algorithm(self, input_data):

        data_dict = dict()
        init_data = copy.deepcopy(input_data)

        mean_list = self.make_mean_value_list(init_data)

        start_index = 'none'
        end_index = 'none'

        for one_dict in mean_list:
            if one_dict['mean_value'] > GLOBAL_THRESHOLD:
                start_index = one_dict['start_index']
            
        for one_dict in reversed(mean_list):
            if one_dict['mean_value'] > GLOBAL_THRESHOLD:
                end_index = one_dict['start_index']
        
        data_dict['data'] = input_data
        data_dict['start_index'] = start_index
        data_dict['end_index'] = end_index

        return data_dict 


    def make_mean_value_list(self, input_data):
        result_list = list()

        data_size = len(input_data)
        range_num = data_size//PREPRO_SHIFT_SIZE

        result_list = [i*PREPRO_SHIFT_SIZE for i in range(range_num)]

        while True:
            temp = result_list[-1]+PREPRO_FRAME_SIZE
            if temp > data_size:
                result_list.pop()
            else:
                break

        result_mean_list = list()

        for idx in result_list:
            temp_dict = dict()

            temp = idx+PREPRO_FRAME_SIZE
            temp_data = input_data[idx:temp]    
            
            mean_value = np.mean(np.abs(temp_data))
            temp_dict['start_index'] = idx
            temp_dict['mean_value'] = mean_value
            result_mean_list.append(temp_dict)

        return result_mean_list 





    
