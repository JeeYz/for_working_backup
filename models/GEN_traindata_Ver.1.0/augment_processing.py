
from numpy import random
from global_variables import *
import global_variables as gv
from CW_class_data import TrainData
import gen_traindata_process as gentrain


def time_stretch_process(files_list):

    for one_file in files_list:
        data = copy.deepcopy(one_file['data'][0])
        for rate in rate_list:
            aug_data = librosa.effects.time_stretch(data, rate)
            one_file['data'].append(aug_data)

    return files_list


def add_zero_padding_back(input_data):
    temp_tail_num = gv.FULL_SIZE - len(input_data)
    zero_padding = np.zeros(temp_tail_num, dtype=gv.TRAIN_DATA_TYPE)
    result_data = np.append(input_data, zero_padding)
    return result_data


def set_initial_dict(init_dict):
    
    init_data = init_dict['data']
    temp_start = init_dict['start_index']
    
    temp_end_of_data = temp_start+gv.FULL_SIZE

    cond = len(init_data) < temp_end_of_data
    if cond:
        temp_data = add_zero_padding_back(init_data[temp_start:])
    else:
        temp = temp_start+gv.FULL_SIZE
        temp_data = init_data[temp_start:temp]

    if len(temp_data) != gv.FULL_SIZE:
        raise Exception("데이터의 길이가 정해진 길이가 아닙니다.")
    
    init_dict['data'] = temp_data
    init_dict['auged_position'] = 0
    init_dict['data_length'] = len(temp_data)
    init_dict['auged_boolean'] = True

    return


def aug_position_process(input_files_list):

    for one_file in input_files_list:
        init_dict = one_file['file_data'][0]

        init_start = init_dict['start_index']
        init_data = init_dict['data']

        for i in range(gv.DATA_AUG_POSITION):
            temp_dict = gv.GLOBAL_CW_TRAINDATA.gen_file_data_dict()

            shift_value_of_start = int((i+1)*gv.BLOCK_OF_RANDOM)
            temp_start = int(init_start-shift_value_of_start)

            if temp_start < 0:
                break
            
            temp_end_of_data = temp_start+gv.FULL_SIZE

            cond = len(init_data) < temp_end_of_data
            if cond:
                temp_data = add_zero_padding_back(init_data[temp_start:])
            else:
                temp = temp_start+gv.FULL_SIZE
                temp_data = init_data[temp_start:temp]

            if len(temp_data) != gv.FULL_SIZE:
                raise Exception("데이터의 길이가 정해진 길이가 아닙니다.")
            
            temp_dict['data'] = temp_data
            temp_dict['auged_position'] = shift_value_of_start
            temp_dict['data_length'] = len(temp_data)
            temp_dict['auged_boolean'] = True
            temp_dict['data_label'] = one_file['file_label']

            one_file['file_data'].append(temp_dict)

        set_initial_dict(init_dict)


def return_block_size(input_dict):

    return block_size, aug_num


def aug_position_process_2(input_files_list):

    for one_file in input_files_list:
        init_dict = copy.deepcopy(one_file['file_data'][0])
        origin_dict = copy.deepcopy(one_file['file_data'][0])

        init_start = init_dict['start_index']
        init_data = init_dict['data']

        block_size, aug_num = return_block_size(init_dict)

        for one_rate in rate_list:
            if one_rate == 1.0:
                origin_start = origin_dict['start_index']
                origin_data = origin_dict['data']

                init_start = origin_start                
                init_data = origin_data
                
            else:
                init_data = librosa.effects.time_stretch(init_data, one_rate)
                para_dict = dict()
                para_dict['data'] = init_data

                return_dict = trigal.signal_trigger_algorithm(para_dict)
                init_start = return_dict['start_index']
                init_data = return_dict['data']

                block_size, aug_num = return_block_size(return_dict)

            for i in range(aug_num):
                temp_dict = gentrain.gen_file_data_dict()

                shift_value_of_start = int((i+1)*block_size)
                temp_start = int(init_start-shift_value_of_start)

                if temp_start < 0:
                    break
                
                temp_end_of_data = temp_start+gv.FULL_SIZE

                cond = len(init_data) < temp_end_of_data
                if cond:
                    temp_data = add_zero_padding_back(init_data[temp_start:])
                else:
                    temp = temp_start+gv.FULL_SIZE
                    temp_data = init_data[temp_start:temp]

                if len(temp_data) != gv.FULL_SIZE:
                    raise Exception("데이터의 길이가 정해진 길이가 아닙니다.")
                
                temp_dict['data'] = temp_data
                temp_dict['auged_position'] = shift_value_of_start
                temp_dict['data_length'] = len(temp_data)
                temp_dict['auged_boolean'] = True
                temp_dict['data_label'] = one_file['file_label']

                one_file['file_data'].append(temp_dict)

            set_initial_dict(init_dict)
            # print('augment complete...'+str(one_file['file_label']))


def random_position_process(input_files_list):

    for one_file in input_files_list:
        init_dict = one_file['file_data'][0]

        init_start = init_dict['start_index']
        init_end = init_dict['end_index']
        init_data = init_dict['data']
        init_gap_size = init_dict['gap_start_end']

        init_random_para = gv.FULL_SIZE-init_gap_size
        random_para = init_random_para//gv.BLOCK_OF_RANDOM
            
        random_value_list = list()

        for i in range(gv.DATA_AUG_POSITION):
            temp_dict = gv.GLOBAL_CW_TRAINDATA.gen_file_data_dict()

            if i == 0:
                random_front_value = 0
            else:
                random_front_value = np.random.randint(random_para)*gv.BLOCK_OF_RANDOM

            if random_front_value in random_value_list:
                continue

            random_value_list.append(random_front_value)

            temp_dict = dict()

            temp_start = init_start-random_front_value
            if temp_start < 0:
                temp_start = 0

            temp_end_of_data = temp_start+gv.FULL_SIZE

            cond = len(init_data) < temp_end_of_data
            if cond:
                temp_data = add_zero_padding_back(init_data[temp_start:])
            else:
                temp = temp_start+gv.FULL_SIZE
                temp_data = init_data[temp_start:temp]

            if len(temp_data) != gv.FULL_SIZE:
                print("initial data information")
                print("init. start : {start}, init. end : {end}, init. data length : {len}, init. gap size : {gap}".format(
                    start=init_start,
                    end=init_end,
                    len=len(init_data),
                    gap=init_gap_size,
                ))
                print("modified data information")
                print("start : {start}, end : {end}, data length : {len}, gap size : {gap}, random front size : {random}".format(
                    start=temp_start,
                    end=temp,
                    len=len(temp_data),
                    gap=temp-temp_start,
                    random=random_front_value,
                ))
                print(one_file['filename'])
                raise Exception("데이터의 길이가 정해진 길이가 아닙니다.")

            temp_dict['data'] = temp_data
            temp_dict['auged_position'] = random_front_value
            temp_dict['data_length'] = len(temp_data)
            temp_dict['auged_boolean'] = True
            temp_dict['label'] = one_file['file_label']

            one_file['file_data'].append(temp_dict)

        set_initial_dict(init_dict)


def random_position_zero_padding(input_files_list):

    exception_num = 0

    for one_file in input_files_list:
        auged_data_list = list()
        for one_data in one_file['data']:

            init_start = one_data['start_index']
            init_end = one_data['end_index']
            init_data = one_data['data']
            init_gap_size = one_data['gap_start_end']

            key_data = init_data[init_start:init_end]
            init_random_para = gv.FULL_SIZE-init_gap_size

            if len(key_data) > gv.FULL_SIZE:
                try:
                    raise Exception("key data 의 크기가 full size를 초과합니다.")
                except Exception as e:
                    print(e, "label : {label}".format(label=one_file['label']))
                    exception_num += 1
                    continue

            if init_random_para < gv.BLOCK_OF_RANDOM:
                # 1회용 코드
                try:
                    raise Exception("랜덤 파라미터의 크기가 작습니다.")
                except Exception as e:
                    print(e, "label : {label}".format(label=one_file['label']))

                    zeros_back = np.zeros(init_random_para , dtype=gv.TRAIN_DATA_TYPE)

                    modification_data = np.append(init_mod_data, zeros_back)
                    auged_data = modification_data[:gv.FULL_SIZE]

                    if len(auged_data) != gv.FULL_SIZE:
                        try:
                            raise Exception("에러발생 1")
                        except Exception as e:
                            print(e)
                        
                    temp_dict['data'] = auged_data
                    temp_dict['position_value'] = -1 
                    temp_dict['data_length'] = len(auged_data)
                    temp_dict['label'] = one_file['label']

                    auged_data_list.append(temp_dict)                    

                    exception_num += 1
                    continue

            temp_zeros = np.zeros(init_random_para, dtype=gv.TRAIN_DATA_TYPE)
            init_mod_data = np.append(key_data, temp_zeros)
            random_para = init_random_para//gv.BLOCK_OF_RANDOM
            
            random_value_list = list()

            for i in range(gv.DATA_AUG_POSITION):
                temp_dict = dict()

                if i == 0:
                    random_front_value = 0
                else:
                    random_front_value = np.random.randint(random_para)*gv.BLOCK_OF_RANDOM

                if random_front_value in random_value_list:
                    continue

                random_value_list.append(random_front_value)

                zeros_front = np.zeros(random_front_value, dtype=gv.TRAIN_DATA_TYPE)

                modification_data = np.append(zeros_front, init_mod_data)
                auged_data = modification_data[:gv.FULL_SIZE]

                if len(auged_data) != gv.FULL_SIZE:
                    try:
                        raise Exception("에러발생 2")
                    except Exception as e:
                        print(e, "raw data length : {len1}, {len2}, {len3}, {len4}".format(
                            len1=len(modification_data), 
                            len2=len(auged_data),
                            len3=len(key_data),
                            len4=init_gap_size,
                        ), init_start, init_end)

                temp_dict['data'] = auged_data
                temp_dict['position_value'] = random_front_value 
                temp_dict['data_length'] = len(auged_data)
                temp_dict['label'] = one_file['file_label']

                auged_data_list.append(temp_dict)                    


        one_file['data'] = auged_data_list

    print("number of exceptions : {num_except}".format(num_except=exception_num))

    return input_files_list



