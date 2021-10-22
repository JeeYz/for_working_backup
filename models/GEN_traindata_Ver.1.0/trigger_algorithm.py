
from global_variables import *
import test_and_check as test


def apply_trigger_algorithm(result_list):

    for one_file in result_list:
        for i,one_data in enumerate(one_file['data']):
            result_dict = signal_trigger_algorithm(one_data, one_file['train'])
            one_file['data'][i] = result_dict

    return result_list


def signal_trigger_algorithm(input_data, train_flag):

    data_dict = dict()
    init_data = copy.deepcopy(input_data)

    mean_list = make_mean_value_list(init_data)

    start_index = 'none'
    end_index = 'none'

    if train_flag is True:
        threshold_ = GLOBAL_THRESHOLD
    else:
        threshold_ = GLOBAL_THRESHOLD_TEST

    for one_dict in mean_list:
        if one_dict['mean_value'] > threshold_:
            start_index = one_dict['start_index']
            break
        
    for one_dict in reversed(mean_list):
        if one_dict['mean_value'] > threshold_:
            end_index = one_dict['start_index'] + PREPRO_FRAME_SIZE
            break

    try:
        a = end_index-start_index
    except TypeError as e:
        print(e)
        print(end_index, start_index)
        test.draw_single_graph(input_data)
    
    if train_flag is False:
        start_index = start_index+5000
        if start_index < 0:
            start_index = 0

    data_dict['data'] = input_data
    data_dict['start_index'] = start_index
    data_dict['end_index'] = end_index
    data_dict['gap_start_end'] = end_index-start_index
    data_dict['data_length'] = len(input_data)

    return data_dict 


def make_mean_value_list(input_data):
    result_list = list()

    data_size = len(input_data)
    range_num = data_size//PREPRO_SHIFT_SIZE

    result_list = [i*PREPRO_SHIFT_SIZE for i in range(range_num)]

    #%% 무한 반복 루프 제거
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






