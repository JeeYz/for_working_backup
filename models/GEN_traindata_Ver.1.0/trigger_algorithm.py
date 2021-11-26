
from numpy.core.fromnumeric import mean
from global_variables import *
import test_and_check as test
# from global_variables import GLOBAL_THRESHOLD
import global_variables as gv


def apply_trigger_algorithm(input_data_list, train_bool):

    for one_file in input_data_list :
        # print(one_file)
        for i,one_data in enumerate(one_file['file_data']):
            signal_trigger_algorithm(one_data, train_bool)


def new_gen_start_end_index(mean_list):
    threshold_ = gv.GLOBAL_THRESHOLD
    
    for one_dict in mean_list:
        temp_val = one_dict['mean_value']*gv.STRONG_COE
        if temp_val > threshold_:
            start_index = one_dict['start_index']
            break
        
    for one_dict in reversed(mean_list):
        temp_val = one_dict['mean_value']*gv.STRONG_COE
        if temp_val > threshold_:
            end_index = one_dict['start_index'] + gv.PREPRO_FRAME_SIZE
            break
        
    
    return start_index, end_index


def add_zero_padding_front(start_index, input_data):
    temp_front_num = -1*start_index
    zero_padding = np.zeros(temp_front_num, dtype=gv.TRAIN_DATA_TYPE)
    result_data = np.append(zero_padding, input_data)
    new_start_index = 0
    return new_start_index, result_data


def add_zero_padding_tail(end_index, input_data):
    temp_tail_num = end_index - len(input_data)
    zero_padding = np.zeros(temp_tail_num, dtype=gv.TRAIN_DATA_TYPE)
    result_data = np.append(input_data, zero_padding)
    new_end_index = len(result_data)
    return new_end_index, result_data


def signal_trigger_algorithm(one_file_data, train_flag):
    input_data = one_file_data['data']

    init_data = copy.deepcopy(input_data)

    mean_list = make_mean_value_list(init_data)

    mean_val_list = list()
    for one in mean_list:
        mean_val_list.append(one['mean_value'])

    start_index = 'none'
    end_index = 'none'

    if train_flag is 'train':
        threshold_ = gv.GLOBAL_THRESHOLD
    else:
        threshold_ = gv.GLOBAL_THRESHOLD_TEST

    for one_dict in mean_list:
        if one_dict['mean_value'] > threshold_:
            start_index = one_dict['start_index']
            break
        
    for one_dict in reversed(mean_list):
        if one_dict['mean_value'] > threshold_:
            end_index = one_dict['start_index'] + gv.PREPRO_FRAME_SIZE
            break

    try:
        start_index = start_index-gv.HEAD_SIZE
    except TypeError as e:
        print(e)
        print("start index 나 end index에 정상 데이터가 입력되지 않았습니다.")
        print(end_index, start_index)
        print(json.dumps(
            mean_list,
            sort_keys=False, 
            indent=4, 
            default=str, 
            ensure_ascii=False
        ))
        test.draw_single_graph(input_data)
        test.draw_single_graph(mean_val_list)
        start_index, end_index = new_gen_start_end_index(mean_list)

    if start_index < 0:
        end_index = end_index-start_index
        start_index, init_data = add_zero_padding_front(start_index, init_data)
        one_file_data['data'] = init_data

    end_index = end_index+gv.TAIL_SIZE
    if end_index > len(init_data):
        end_index, init_data = add_zero_padding_tail(end_index, init_data)
        one_file_data['data'] = init_data

    gap_temp = end_index-start_index
    
    one_file_data['start_index'] = start_index
    one_file_data['end_index'] = end_index
    one_file_data['gap_start_end'] = gap_temp
    one_file_data['data_length'] = len(init_data)


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

    
def add_zero_padding(input_data):
    padding_size = gv.FULL_SIZE//2
    zero_padding = np.zeros(padding_size, dtype=gv.TRAIN_DATA_TYPE)
    result_data = np.append(zero_padding, input_data)
    result = np.append(result_data, zero_padding)

    return result


def signal_trigger_algorithm_for_decode(input_data):
    # print(gv.GLOBAL_THRESHOLD)
    init_data = copy.deepcopy(input_data)

    init_data = add_zero_padding(init_data)

    mean_list = make_mean_value_list(init_data)

    start_index = 'none'
    end_index = 'none'

    threshold_ = gv.GLOBAL_THRESHOLD

    for one_dict in mean_list:
        if one_dict['mean_value'] > threshold_:
            temp = one_dict['start_index']
            if temp < gv.TARGET_SAMPLE_RATE:
                continue
            start_index = one_dict['start_index']
            break

    if start_index == 'none'  :
        return None

    start_index = start_index-gv.DECODING_FRONT_SIZE
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


