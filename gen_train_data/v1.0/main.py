

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


def print_json_dump(input_data_list):


    print(json.dumps(
                    input_data_list[0], 
                    sort_keys=False, 
                    indent=4, 
                    default=str, 
                    ensure_ascii=False
    ))
    print(json.dumps(
                    input_data_list[-1], 
                    sort_keys=False, 
                    indent=4, 
                    default=str, 
                    ensure_ascii=False
    ))


    return



def main():
    print(label_dict)

    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # make list of dicts for each data
    zeroth_none_data_list = fpro.run_processing_data(none_data_path)
    train_data_list = fpro.run_processing_data(train_data_path)
    test_data_list = fpro.run_processing_data(test_data_path)

    # print volume of data
    print("\n")
    print("zeroth data volume : {num}".format(num=len(zeroth_none_data_list)))
    print("train data volume : {num}".format(num=len(train_data_list)))
    print("test data volume : {num}".format(num=len(test_data_list)))
    print("\n")


    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    train_data_list = spro.detect_saturation_signal(train_data_list)   
    train_data_list = spro.detect_saturation_signal(train_data_list)   

    train_data_list = spro.read_each_files_to_data(train_data_list)

    tcheck.check_number_of_labels(train_data_list)


    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    zeroth_none_data_list = spro.detect_saturation_signal(zeroth_none_data_list)   
    zeroth_none_data_list = spro.detect_saturation_signal(zeroth_none_data_list)   

    zeroth_none_data_list = spro.read_each_files_to_data(zeroth_none_data_list)

    tcheck.check_number_of_labels(zeroth_none_data_list)


    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    test_data_list = spro.detect_saturation_signal(test_data_list)   
    test_data_list = spro.detect_saturation_signal(test_data_list)   

    test_data_list = spro.read_each_files_to_data(test_data_list)

    tcheck.check_number_of_labels(test_data_list)


    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    train_data_list = augp.time_stretch_process(train_data_list)

    train_data_list = spro.standardize_data(train_data_list)

    print_json_dump(train_data_list)

    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    zeroth_none_data_list = spro.standardize_data(zeroth_none_data_list)

    print_json_dump(zeroth_none_data_list)


    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    test_data_list = spro.standardize_data(test_data_list)

    print_json_dump(test_data_list)


    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    train_data_list = trigal.apply_trigger_algorithm(train_data_list)

    train_data_list = moddi.modifying_start_end_indexs(train_data_list)

    print_json_dump(train_data_list)

    tcheck.check_data_gap_size(train_data_list)

    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    zeroth_none_data_list = trigal.apply_trigger_algorithm(zeroth_none_data_list)

    zeroth_none_data_list = moddi.modifying_start_end_indexs(zeroth_none_data_list)

    print_json_dump(zeroth_none_data_list)

    tcheck.check_data_gap_size(zeroth_none_data_list)

    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    test_data_list = trigal.apply_trigger_algorithm(test_data_list)

    test_data_list = moddi.modifying_start_end_indexs(test_data_list)

    print_json_dump(test_data_list)

    tcheck.check_data_gap_size(test_data_list)


    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    train_data_list = augp.aug_position_process(train_data_list) 
    print_json_dump(train_data_list)

    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    zeroth_none_data_list = nonepro.none_aug_process(zeroth_none_data_list)
    print_json_dump(zeroth_none_data_list)
    
    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    test_data_list = nonepro.none_aug_process(test_data_list )
    print_json_dump(test_data_list)


    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    tcheck.check_data_length(train_data_list)
    tcheck.check_data_length(zeroth_none_data_list)
    tcheck.check_data_length(test_data_list)



    #%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if INCLUDE_ZEROTH_NONE is True:
        data_list = [
            train_data_list,
            zeroth_none_data_list,
            test_data_list,
        ]
    else:
        data_list = [
            train_data_list,
            test_data_list,
        ]
    
    gendata.write_numpy(data_list)


    return
    










if __name__ == '__main__':
    main()



