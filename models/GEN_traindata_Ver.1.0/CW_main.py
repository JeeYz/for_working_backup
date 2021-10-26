
from json import decoder
from scipy import signal
from global_variables import *
import file_processing as fpro
import signal_processing as spro
import test_and_check as tcheck
import augment_processing as augp
import trigger_algorithm as trigal
import modifying_data_and_info as moddi
import none_aug_processing as nonepro
import gen_data_files as gendata
import CW_json_files_decoder as decoder
import CW_signal_processing as cwsig


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

    # print volume of data
    # GLOBAL_CW_TRAINDATA.print_whole_train_data_info()
    GLOBAL_CW_TRAINDATA.print_whole_data_length()

    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    cw_traindata_list = cwsig.receive_files_list()
    print_json_dump(cw_traindata_list)

    raise Exception("프로그램을 종료합니다.")

    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    cw_traindata_list = trigal.apply_trigger_algorithm(cw_traindata_list)
    cw_traindata_list = moddi.modifying_start_end_indexs(cw_traindata_list)

    print_json_dump(cw_traindata_list)

    tcheck.check_data_gap_size(cw_traindata_list)
    
    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    cw_traindata_list = augp.aug_position_process(cw_traindata_list) 
    print_json_dump(cw_traindata_list)
    
    tcheck.check_data_length(cw_traindata_list)
    
    gendata.write_numpy(cw_traindata_list)

    return
    










if __name__ == '__main__':
    main()



