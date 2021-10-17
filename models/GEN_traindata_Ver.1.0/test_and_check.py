

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




