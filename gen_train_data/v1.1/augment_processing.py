from global_variables import *
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
                pass
                
            else:
                mod_by_time_stretch_data = librosa.effects.time_stretch(init_data, one_rate)
                para_dict = set_initial_dict()
                para_dict['data'] = mod_by_time_stretch_data

                gv.triga.signal_trigger_algorithm_with_middle_index(para_dict)

                para_dict['auged_boolean'] = True
                para_dict['data_label'] = one_file['file_label']

                one_file['file_data'].append(para_dict)

    
    return






