
from global_variables import *
import trigger_algorithm as trig


#%%
class Signal_Processing():
    def __init__(self):
        pass


    def read_each_files_to_data(self, files_list):

        for one_file in files_list:
            filename = one_file['filename']
            sr, data = wavfile.read(filename)
            data = np.array(data, dtype=TRAIN_DATA_TYPE)

            one_file['data'] = list()
            one_file['data'].append(data)

        return files_list


    def detect_saturation_signal(self, files_list):

        num = 0
        a = len(files_list)
        print("start detecting saturation signals...")
        print("input data length : {length}".format(length=a))
        for one_file in reversed(files_list):
            filename = one_file['filename']
            sr, data = wavfile.read(filename)

            max_value = np.max(data)
            min_value = np.min(data)

            cond1 = max_value < MAX_SIGNAL_VALUE
            cond2 = min_value > MIN_SIGNAL_VALUE
            if cond1 and cond2:
                pass
            else:
                idx_num = files_list.index(one_file)
                del files_list[idx_num]
                num += 1
            
        a = len(files_list)
        print("output data length : {length}".format(length=a))
        print("{dnum} data is deleted...".format(dnum=num))
        print('\n')

        return files_list


    def standardize_data(self, files_list):
        for one_data in files_list:
            if one_data['data'] is not []:
                for i, one_aug in enumerate(one_data['data']):
                    tmp_data = copy.deepcopy(one_aug)
                    one_data['data'][i] = (tmp_data-np.mean(tmp_data))/np.std(tmp_data)

        return files_list

    
    def normalize_data(self, files_list):
        for one_data in files_list:
            if one_data['data'] is not []:
                for i, one_aug in enumerate(one_data['data']):
                    tmp_data = copy.deepcopy(one_aug)
                    min_val = np.min(tmp_data)
                    max_val = np.max(tmp_data)
                    one_data ['data'][i] = ((tmp_data-min_val)/(max_val-min_val)-0.5)*2

        return files_list


    def simple_norm_data(self, files_list):

        for one_data in files_list:
            if one_data['data'] is not []:
                for i, one_aug in enumerate(one_data['data']):
                    tmp_data = copy.deepcopy(one_aug)
                    one_data ['data'][i] = np.array(tmp_data, dtype=TRAIN_DATA_TYPE)/MAX_SIGNAL_VALUE

        return files_list


    def cut_signal(self, files_list):

        for one_data in files_list:
            if one_data['data'] is not []:
                for i, one in enumerate(one_data['data']):
                    tmp_data = copy.deepcopy(one)
                    # trigger input -> module -> output
                    

        return files_list



    





