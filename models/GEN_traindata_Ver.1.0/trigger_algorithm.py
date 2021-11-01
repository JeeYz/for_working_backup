
from global_variables import *
import test_and_check as test
# from global_variables import GLOBAL_THRESHOLD
import global_variables as gv


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
    # range_num = data_size//200
    range_num = data_size//gv.PREPRO_SHIFT_SIZE

    # result_list = [i*200 for i in range(range_num)]
    result_list = [i*gv.PREPRO_SHIFT_SIZE for i in range(range_num)]

    #%% 무한 반복 루프 제거
    while True:
        temp = result_list[-1]+gv.PREPRO_FRAME_SIZE
        # temp = result_list[-1]+400
        if temp > data_size:
            result_list.pop()
        else:
            break

    result_mean_list = list()

    for idx in result_list:
        temp_dict = dict()

        temp = idx+gv.PREPRO_FRAME_SIZE
        # temp = idx+400
        temp_data = input_data[idx:temp]    
        
        mean_value = np.mean(np.abs(temp_data))
        temp_dict['start_index'] = idx
        temp_dict['mean_value'] = mean_value

        result_mean_list.append(temp_dict)

    return result_mean_list 


def fit_fullsize(input_data, end_index):
    temp = end_index-len(input_data)
    tail = np.zeros(temp, dtype=gv.TRAIN_DATA_TYPE)
    # tail = np.zeros(temp, dtype=np.float32)
    result = np.append(input_data, tail)
    return result


def signal_trigger_algorithm_for_decode(input_data):
    # print(gv.GLOBAL_THRESHOLD)
    init_data = copy.deepcopy(input_data)

    mean_list = make_mean_value_list(init_data)

    start_index = 'none'
    end_index = 'none'

    threshold_ = gv.GLOBAL_THRESHOLD
    # threshold_ = 3.0

    for one_dict in mean_list:
        if one_dict['mean_value'] > threshold_:
            start_index = one_dict['start_index']
            break

    if start_index == 'none'  :
        return None

    start_index = start_index-gv.BUFFER_SIZE 
    if start_index < 0:
        start_index = 0

    end_index = start_index+gv.FULL_SIZE
    # end_index = start_index+40000

    if end_index > len(init_data):
        init_data = fit_fullsize(init_data, end_index)

    result = init_data[start_index:end_index]

    return result



if __name__ == '__main__':
    print(FULL_SIZE)
    print(PREPRO_FRAME_SIZE)


