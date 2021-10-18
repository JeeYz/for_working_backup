

from global_variables import *


class Monitoring_Check():
    def __init__(self):
        pass

    
    def check_number_of_labels(self, files_list):

        temp_list = list()
        for one_file in files_list:
            one_label = one_file['label']
            if one_label not in temp_list: 
                temp_list.append(one_label)

        temp_dict = dict()
        for one in temp_list:
            temp_dict[one] = 0
        
        for one_file in files_list:
            one_label = one_file['label']
            temp_dict[one_label]+=1

        print("**numbers of labels**")
        for one in temp_dict.keys():
            print("{label} : {num}".format(label=one, num=temp_dict[one]))
        
        print("\n")

        return


    def check_data_gap_size(self, input_data_list):

        under_full_size_num  = 0
        over_full_size_num = 0

        print_result_list = list()

        for one_file in input_data_list:
            for one_data in one_file['data']:
                if one_data['gap_start_end'] <= FULL_SIZE:
                    under_full_size_num += 1
                else:
                    over_full_size_num += 1
                    if one_file not in print_result_list:
                        print_result_list.append(one_file)


        print("***number of data size***")
        print("number of under full size : {under}".format(under=under_full_size_num))
        print("number of over full size : {over}".format(over=over_full_size_num))
        # print(print_result_list)

        result_dict = dict()
        for one_file in print_result_list:
            temp_label_name = one_file['label']
            for one_data in one_file['data']:
                if one_file['label'] in result_dict.keys():
                    result_dict[temp_label_name] += 1
                else:
                    result_dict[temp_label_name] = 0

        print("\n")
        print("*number of data over full size for each labels")
        for one_label in result_dict.keys():
            print("{label} : {data_num}".format(label=one_label, data_num=result_dict[one_label]))

        print('\n')

        return




