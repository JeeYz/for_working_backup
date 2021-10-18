

from scipy import signal
from global_variables import *
import file_processing as fpro
import signal_processing as spro
import test_and_check as tcheck
import augment_processing as augp
import trigger_algorithm as trigal
import modifying_data_and_info as moddi


def main():
    print(label_dict)

    process_files_class = fpro.File_Processing()

    # make list of dicts for each data
    zeroth_none_data_list = process_files_class.run_processing_data(none_data_path)
    train_data_list = process_files_class.run_processing_data(train_data_path)
    test_data_list = process_files_class.run_processing_data(test_data_path)

    # print volume of data
    print("\n")
    print("zeroth data volume : {num}".format(num=len(zeroth_none_data_list)))
    print("train data volume : {num}".format(num=len(train_data_list)))
    print("test data volume : {num}".format(num=len(test_data_list)))
    print("\n")

    #%% ###########################################################
    signal_process_class = spro.Signal_Processing()

    train_data_list = signal_process_class.detect_saturation_signal(train_data_list)   
    train_data_list = signal_process_class.detect_saturation_signal(train_data_list)   

    train_data_list = signal_process_class.read_each_files_to_data(train_data_list)

    monitoring = tcheck.Monitoring_Check()
    monitoring.check_number_of_labels(train_data_list)

    #%% ###########################################################
    augment_process_class = augp.Augment_Process()

    train_data_list = augment_process_class.time_stretch_process(train_data_list)

    train_data_list = signal_process_class.standardize_data(train_data_list)

    print(json.dumps(
                    train_data_list[0], 
                    sort_keys=False, 
                    indent=4, 
                    default=str, 
                    ensure_ascii=False
    ))
    print(json.dumps(
                    train_data_list[-1], 
                    sort_keys=False, 
                    indent=4, 
                    default=str, 
                    ensure_ascii=False
    ))

    #%% ###########################################################
    trig_class = trigal.Signal_Trigger()
    train_data_list = trig_class.apply_trigger_algorithm(train_data_list)

    train_data_list = moddi.modifying_start_end_indexs(train_data_list)

    print(json.dumps(
                    train_data_list[0], 
                    sort_keys=False, 
                    indent=4, 
                    default=str, 
                    ensure_ascii=False
    ))
    print(json.dumps(
                    train_data_list[-1], 
                    sort_keys=False, 
                    indent=4, 
                    default=str, 
                    ensure_ascii=False
    ))

    monitoring.check_data_gap_size(train_data_list)
    #%% ###########################################################



    return
    



if __name__ == '__main__':
    main()



