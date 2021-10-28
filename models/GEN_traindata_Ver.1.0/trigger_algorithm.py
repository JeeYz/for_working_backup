
from global_variables import *
import test_and_check as test


def apply_trigger_algorithm(input_data_list, train_bool):

    for one_file in input_data_list :
        for i,one_data in enumerate(one_file['file_data']):
            signal_trigger_algorithm(one_data, train_bool)


def signal_trigger_algorithm(one_file_data, train_flag):
    input_data = one_file_data['data']

    init_data = copy.deepcopy(input_data)

    mean_list = make_mean_value_list(init_data)

    start_index = 'none'
    end_index = 'none'

    if train_flag is 'train':
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
    
    one_file_data['start_index'] = start_index
    one_file_data['end_index'] = end_index
    one_file_data['gap_start_end'] = end_index-start_index
    one_file_data['data_length'] = len(input_data)


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


def fit_fullsize(input_data, end_index):
    temp = end_index-len(input_data)
    tail = np.zeros(temp, dtype=TRAIN_DATA_TYPE)
    result = np.append(input_data, tail)
    return result


def signal_trigger_algorithm_for_decode(input_data):
    init_data = copy.deepcopy(input_data)

    mean_list = make_mean_value_list(init_data)

    start_index = 'none'
    end_index = 'none'

    threshold_ = GLOBAL_THRESHOLD

    for one_dict in mean_list:
        if one_dict['mean_value'] > threshold_:
            start_index = one_dict['start_index']
            break
        
    try:
        a = end_index-start_index
    except TypeError as e:
        print(e)
        print(end_index, start_index)
        test.draw_single_graph(input_data)
    
    end_index = start_index+FULL_SIZE

    if end_index > len(init_data):
        init_data = fit_fullsize(init_data, end_index)

    result = init_data[start_index:end_index]

    return result






