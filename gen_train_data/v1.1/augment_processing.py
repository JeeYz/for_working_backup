import global_variables as gv



def set_initial_dict():
    return_dict = dict()
    
    return_dict['data'] = None 
    return_dict['start_index'] = None
    return_dict['end_index'] = None
    return_dict['auged_position'] = None
    return_dict['data_length'] = None
    return_dict['auged_boolean'] = None

    return return_dict



def check_gap_start_end(input_dict):
    init_start = input_dict['start_index']
    init_end = input_dict['end_index']
    init_gap = input_dict['gap_start_end']

    if (init_end-init_start) != init_gap:
        init_gap = init_end-init_start
        input_dict['gap_start_end'] = init_gap



def aug_only_time_stretch(input_files_list):

    for one_file in input_files_list:
        init_dict = copy.deepcopy(one_file['file_data'][0])
        origin_dict = copy.deepcopy(one_file['file_data'][0])

        check_gap_start_end(init_dict)
        check_gap_start_end(origin_dict)

        init_start = init_dict['start_index']
        init_data = init_dict['data']

        for one_rate in gv.rate_list:
            if one_rate == 1.0:
                origin_start = origin_dict['start_index']
                origin_data = origin_dict['data']

                init_start = origin_start                
                init_data = origin_data
                
            else:
                init_data = librosa.effects.time_stretch(init_data, one_rate)
                para_dict = dict()
                para_dict['data'] = init_data

                gv.trigal.signal_trigger_algorithm(para_dict, gv.TEMP_FLAG)
                init_start = para_dict['start_index']
                init_data = para_dict['data']

                block_size, aug_num = return_block_size(para_dict)

            if 'noncmd_' in one_file['filename']:
                aug_num = 2
                block_size = block_size*gv.DATA_AUG_POSITION//2
                # continue

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
                    temp_end_of_data = temp_start+gv.FULL_SIZE
                    temp_data = init_data[temp_start:temp_end_of_data]

                if len(temp_data) != gv.FULL_SIZE:
                    raise Exception("데이터의 길이가 정해진 길이가 아닙니다.")

                temp_dict['auged_position'] = shift_value_of_start
                temp_dict['data_length'] = len(temp_data)
                temp_dict['auged_boolean'] = True
                temp_dict['data_label'] = one_file['file_label']
                temp_dict['data'] = temp_data

                one_file['file_data'].append(temp_dict)

        set_initial_dict(init_dict)
        one_file['file_data'].append(init_dict)
    
        del one_file['file_data'][0]
    
    return






