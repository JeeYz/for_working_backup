from global_variables import *
import global_variables as gv



def apply_trigger_algorithm(input_data_list):

    for one_file in input_data_list :
        # print(one_file)
        for i,one_data in enumerate(one_file['file_data']):
            signal_trigger_algorithm_with_middle_index(one_data)




def new_gen_start_end_index(mean_list):
    threshold_ = gv.GLOBAL_THRESHOLD

    max_value = 0
    max_index = 0

    for one_dict in mean_list:
        if max_value < one_dict['mean_value']:
            max_value = one_dict['mean_value']
            max_index = one_dict['start_index']
    
    start_index = max_index
    end_index = max_index
    
    return start_index, end_index



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



    
def add_zero_padding(input_data):
    padding_size = gv.FULL_SIZE//2
    zero_padding = np.zeros(padding_size, dtype=gv.TRAIN_DATA_TYPE)
    result_data = np.append(zero_padding, input_data)
    result = np.append(result_data, zero_padding)

    return result



def signal_trigger_algorithm_with_middle_index(one_file_data):
    input_data = one_file_data['data']

    init_data = copy.deepcopy(input_data)

    mean_list = make_mean_value_list(init_data)

    mean_val_list = list()
    for one in mean_list:
        mean_val_list.append(one['mean_value'])

    start_index = 'none'
    end_index = 'none'

    threshold_ = gv.GLOBAL_THRESHOLD

    for one_dict in mean_list:
        if one_dict['mean_value'] > threshold_:
            start_index = one_dict['start_index']
            break
        
    for one_dict in reversed(mean_list):
        if one_dict['mean_value'] > threshold_:
            end_index = one_dict['start_index']
            break

    try:
        if start_index == 'none' or end_index =='none':
            raise Exception("start index 나 end index에 정상 데이터가 입력되지 않았습니다.")
    except Exception as e:
        print(end_index, start_index)
        print("정상적인 start index와 end index를 재반환합니다.")
        start_index, end_index = new_gen_start_end_index(mean_list)
        print(end_index, start_index)

    half_of_fullsize = gv.FULL_SIZE//2

    add_zero_padding_data = add_zero_padding(init_data)

    start_index = start_index + half_of_fullsize
    end_index = end_index + half_of_fullsize

    gap_temp = end_index-start_index

    middle_index = (start_index + end_index)//2

    new_start_index = half_of_fullsize - (middle_index - start_index) 
    new_end_index = new_start_index + gap_temp

    new_start_index = new_start_index - gv.PREPRO_FRAME_SIZE
    new_end_index = new_end_index + gv.PREPRO_FRAME_SIZE

    new_data = add_zero_padding_data[(middle_index-half_of_fullsize):(middle_index+half_of_fullsize)]
    
    one_file_data['start_index'] = new_start_index
    one_file_data['end_index'] = new_end_index
    one_file_data['gap_start_end'] = gap_temp
    one_file_data['data_length'] = len(init_data)
    one_file_data['data'] = new_data





if __name__ == '__main__':
    print(FULL_SIZE)
    print(PREPRO_FRAME_SIZE)


