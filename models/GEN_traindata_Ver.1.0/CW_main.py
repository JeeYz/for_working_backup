
from json import decoder
from scipy import signal
from global_variables import *


def print_json_dump(target_dict):
    print(json.dumps(
                    target_dict,
                    sort_keys=False, 
                    indent=4, 
                    default=str, 
                    ensure_ascii=False
    ))

    return


def main():
    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    decoder.json_read_main()
    GLOBAL_CW_TRAINDATA.print_whole_data_length()

    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    whole_data_list = GLOBAL_CW_TRAINDATA.get_whole_data_list()
    cwsig.receive_files_list(whole_data_list )

    GLOBAL_CW_TRAINDATA.print_whole_data_length()

    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    whole_data_list = GLOBAL_CW_TRAINDATA.get_whole_data_list()
    train_bool = GLOBAL_CW_TRAINDATA.get_traindata()['kind_of_data']
    trigal.apply_trigger_algorithm(whole_data_list, train_bool)
    GLOBAL_CW_TRAINDATA.print_whole_train_data_info()
    GLOBAL_CW_TRAINDATA.print_whole_data_length()

    # temp = input("stop at check point...")

    whole_data_list = GLOBAL_CW_TRAINDATA.get_whole_data_list()
    moddi.modifying_start_end_indexs(whole_data_list)
    GLOBAL_CW_TRAINDATA.print_whole_train_data_info()
    GLOBAL_CW_TRAINDATA.print_whole_data_length()

    whole_data_list = GLOBAL_CW_TRAINDATA.get_whole_data_list()
    tcheck.check_data_gap_size(whole_data_list)
    
    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    whole_data_list = GLOBAL_CW_TRAINDATA.get_whole_data_list()

    if AUGMENT_FLAG is 0:
        moddi.make_fit_full_size_data(whole_data_list)
        GLOBAL_CW_TRAINDATA.print_whole_train_data_info()
        GLOBAL_CW_TRAINDATA.print_whole_data_length()
    elif AUGMENT_FLAG is 1:
        augp.aug_position_process(whole_data_list) 
        GLOBAL_CW_TRAINDATA.print_whole_train_data_info()
        GLOBAL_CW_TRAINDATA.print_whole_data_length()
    elif AUGMENT_FLAG is 2:
        augp.random_position_process(whole_data_list) 
        GLOBAL_CW_TRAINDATA.print_whole_train_data_info()
        GLOBAL_CW_TRAINDATA.print_whole_data_length()

    tcheck.check_data_length(whole_data_list)
    
    GLOBAL_CW_TRAINDATA.generate_numpy_file()

    return
    










if __name__ == '__main__':
    main()



