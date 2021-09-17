import os
from types import prepare_class
from numpy.core.numeric import full
from scipy.io import wavfile
import numpy as np
import json
import librosa

import time
import copy
import matplotlib.pyplot as plt


FULL_SIZE = 20000


def main():
    gen_data = control_gen_data()
    gen_data.generate_data()

    return



#%%
class control_gen_data():
    def __init__(self):
        self.random_rate = 0.1

    def generate_data(self):
        prepro = Preprocessing_data()
        result_data = prepro.preprocess_data()
        print(len(result_data))

        # for one in result_data:
        #     print(type(one))

        aug_data_pro = Augment_data()
        return_data = aug_data_pro.read_data_list(result_data)
        print(len(return_data))


    def print_plt_data(self):
        pass

    def write_numpy_file(self):
        pass



#%%
class Preprocessing_data():
    def __init__(self):
        self.full_size = FULL_SIZE
        self.threshold_rate = 0.15
        self.noise_threshold_rate = 0.05

        self.frame_size = 1000
        self.shift_size = 500

        self.label_list = [
                'camera',
                'picture',
                'record',
                'stop',
                'end',
                ]


    def preprocess_data(self):
        
        result_list = self.find_train_data_files(
            "D:\\voice_data_backup\\zeroth_none_label",
            ".wav"
            )
        none_data_dict = self.make_data_dict(result_list)
        
        # print_contents_of_dict(none_data_dict)
        # get_max_value(none_data_dict)

        none_label_data = self.gen_none_label_data(none_data_dict)

        print(len(none_label_data))

        files_list = self.find_train_data_files(
                "D:\\voice_data_backup\\PNC_DB_ALL",
                ".wav"
                )

        # print(len(files_list))

        result_data_list = self.select_data_for_saturation(files_list)

        # print(len(result_data_list))

        train_label_data_dict = self.make_data_dict(result_data_list)
        result_train_data_list = self.cut_wav_files(train_label_data_dict)

        result_train_data = list()

        result_train_data.extend(none_label_data)
        result_train_data.extend(result_train_data_list)

        print(len(result_train_data_list))
        print('complete~!!')

        return result_train_data


    def add_head_and_tail_none(self, input_data):

        tmp_front_num = (self.full_size - len(input_data))//2
        tmp_tail_num = self.full_size - len(input_data) - tmp_front_num

        low_val = np.min(input_data)*self.noise_threshold_rate
        high_val = np.max(input_data)*self.noise_threshold_rate
        
        front_random_data = np.random.randint(low_val, high_val, tmp_front_num)
        tail_random_data = np.random.randint(low_val, high_val, tmp_tail_num)

        tmp = np.append(front_random_data, input_data)
        output_data = np.append(tmp, tail_random_data)

        return output_data


    def gen_none_label_data(self, data_dict):

        none_data_list = list()

        for one_key in data_dict.keys():

            tmp_data = data_dict[one_key]
            tmp_len = len(tmp_data)

            tmp_range = int(self.full_size*1.5)

            if tmp_len > tmp_range:
                tmp = tmp_data[:self.full_size]
                mid_point = -1

                temp_dict = dict()
                temp_dict['data'] = tmp
                temp_dict['label'] = 'none'
                temp_dict['mid_val'] = mid_point

                none_data_list.append(temp_dict)

                tmp_num = -1*self.full_size
                tmp = tmp_data[tmp_num:]
                mid_point = -1

            elif tmp_range >= tmp_len\
                        and\
                    tmp_len > self.full_size:
                tmp = tmp_data[:self.full_size]
                mid_point = -1

            else:
                mid_point = self.full_size//2
                tmp = self.add_head_and_tail_none(tmp_data)
            
            temp_dict = dict()
            temp_dict['data'] = tmp
            temp_dict['label'] = 'none'
            temp_dict['mid_val'] = mid_point

            try:
                none_data_list.append(temp_dict)
            except len(tmp) != self.full_size:
                print("occured exception...")
                continue

        return none_data_list


    def get_max_value(self, data_dict):

        max_val = 0

        for one in data_dict.keys():
            temp = data_dict[one]
            temp_val = len(temp)
            if max_val < temp_val:
                max_val = temp_val

        print("max_val : {}".format(max_val))

        return



    def print_contents_of_dict(self, tmp_dict):

        print(json.dumps(tmp_dict, sort_keys=False, indent=4, default=str, ensure_ascii=False))

        return


    def make_data_dict(self, files_list):

        train_data_dict = dict()

        for one_file in files_list:
            # w_sr, w_data = wavfile.read(one_file)
            try:
                w_sr, w_data = wavfile.read(one_file)
            except wavfile.WavFileWarning:
                print("\noccur warning...\n")
                continue
            except RuntimeWarning:
                print("\noccur warning...\n")
                continue
            # except:
            #     print("\noccur warning...\n")
            #     continue


            train_data_dict[one_file] = w_data

        return train_data_dict


    def find_train_data_files(self, files_path, file_ext):

        all_data_file = list()

        for (path, dir, files) in os.walk(files_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == file_ext:
                    file_name = path + "\\" + filename
                    all_data_file.append(file_name)

        return all_data_file


    def add_label_info(self, data, filename, mid_point):

        label_name = filename.split("\\")[-2]

        temp_dict = dict()

        temp_dict["data"] = data
        temp_dict['label'] = label_name
        temp_dict['mid_val'] = mid_point

        return temp_dict


    def make_frame_list(self, data_size):

        result_list = list()

        tmp = data_size//self.shift_size-1

        for i in range(tmp):
            result_list.append(i*self.shift_size)

        return result_list


    def cut_one_wav_file(self, input_data):

        abs_data = np.abs(input_data)
        max_mag = np.max(abs_data)
        threshold_val = max_mag*self.threshold_rate

        frame_start_list = self.make_frame_list(len(input_data))

        start_idx = 0
        end_idx = 0

        for i in frame_start_list:
            tmp_end = i+self.frame_size
            tmp_data = input_data[i:tmp_end]
            tmp_data = np.abs(tmp_data)
            meanval_tmp_data = np.mean(tmp_data)
            if threshold_val < meanval_tmp_data:
                start_idx = i
                break

        for j in reversed(frame_start_list):
            tmp_end = i+self.frame_size
            tmp_data = input_data[i:tmp_end]
            tmp_data = np.abs(tmp_data)
            meanval_tmp_data = np.mean(tmp_data)
            if threshold_val < meanval_tmp_data:
                end_idx = j+self.frame_size
                break

        one_data_dict = dict()

        # print("***************", start_idx, end_idx)
        one_data_dict['data'] = input_data
        one_data_dict['start'] = start_idx
        one_data_dict['end'] = end_idx

        return one_data_dict


    def modifying_data(self, input_dict):

        tmp_start = input_dict['start']
        tmp_end = input_dict['end']
        tmp_data = input_dict['data']

        low_val = np.min(tmp_data)*self.noise_threshold_rate
        high_val = np.max(tmp_data)*self.noise_threshold_rate

        half_fullsize = self.full_size//2
        mid_point = (tmp_end-tmp_start)//2 + half_fullsize
        # print(len(tmp_data), mid_point, tmp_start, tmp_end)

        tmp_fdata = np.random.randint(low_val, high_val, half_fullsize)
        tmp_edata = np.random.randint(low_val, high_val, half_fullsize)

        tmp_data = np.append(tmp_fdata, tmp_data)
        tmp_data = np.append(tmp_data, tmp_edata)

        new_start = mid_point - half_fullsize
        new_end = mid_point + half_fullsize
        
        tmp_main_data = tmp_data[new_start:new_end]

        if len(tmp_main_data) != self.full_size:
            print("this data is incorrent size...")
            print(new_start, new_end)
            print(len(tmp_main_data), len(tmp_data))
            return "None"
        else:
            return tmp_main_data, (mid_point-new_start)
      

    def cut_wav_files(self, data_dict):

        result_data = list()

        for one_key in data_dict.keys():

            tmp_data = data_dict[one_key]
            
            output_data_dict = self.cut_one_wav_file(tmp_data)
            one_tmp_data, mid_point = self.modifying_data(output_data_dict)
            if one_tmp_data != 'None':
                output_dict = self.add_label_info(
                        one_tmp_data, one_key, mid_point
                        )
                result_data.append(output_dict)
        
        return result_data


    def determine_condition(self, file_name):

        try:
            sample_rate, data = wavfile.read(file_name)
        except wavfile.WavFileWarning:
            print("\noccur warning...\n")
            return 0
        except RuntimeWarning:
            print("\noccur warning...\n")
            return 0
        # except:
        #     print("\noccur warning...\n")
        #     return 0

        max_value = np.max(data)
        min_value = np.min(data)

        # 조건 0 and 1
        if max_value != 32767   \
                    and         \
                min_value != -32768:                        

            return 1
        else:
            return 0


    def select_data_for_saturation(self, files_list):

        result_list = list()

        for one_file in files_list:
            cond = self.determine_condition(one_file)
            if cond == 1:
                result_list.append(one_file)

        return result_list



#%%
class Augment_data():
    def __init__(self):

        self.full_size = FULL_SIZE
        self.augmented_data = list()
        self.time_stretch_list = [
                # 0.9,
                1.0,
                1.1,
                ]
        self.position_aug_frame = 1000

        self.threshold_rate = 0.15
        self.noise_threshold_rate = 0.05

        self.para_num = 0


    def add_noise_data(self, input_data, asize):

        low_val = np.min(input_data)*self.noise_threshold_rate
        high_val = np.max(input_data)*self.noise_threshold_rate

        front_random_data = np.random.randint(low_val, high_val, asize)

        return front_random_data


    def gen_aug_data(self, data_dict, one_data):

        new_data_list = list()

        init_start_idx = data_dict['start'][0]
        init_data = copy.deepcopy(data_dict['data'])
        init_data = np.array(init_data, dtype=np.float64)
        init_label = one_data['label']

        for_cond = len(data_dict['start'])

        for r in self.time_stretch_list:

            aug_data = librosa.effects.time_stretch(init_data, r)

            for one_st in range(for_cond):
                temp = dict()

                cond1 = one_st - init_start_idx
                add_size = copy.deepcopy(cond1)

                if add_size != 0:
                    add_size = np.abs(add_size)
                    return_noise = self.add_noise_data(init_data, add_size)
                    return_noise = np.array(return_noise, dtype=np.float64)
                    if cond1 > 0:
                        temp_data = np.append(return_noise, aug_data)
                    else:
                        temp_data = np.append(aug_data, return_noise)

                    new_data = temp_data[0: self.full_size]
                else:
                    new_data = copy.deepcopy(init_data)
                
                temp['data'] = new_data
                temp['label'] = init_label

                new_data_list.append(temp)
                self.para_num += 1
                print("{} th data generated...".format(self.para_num), end='\r')

        return new_data_list


    def make_aug_list(self, data_dict):

        start_list = list()
        end_list = list()

        tmp_start = data_dict['start']
        tmp_end = data_dict['end']

        start_list.append(tmp_start)
        end_list.append(tmp_end)

        gap_start_end = tmp_end - tmp_start

        temp = copy.deepcopy(tmp_start)
        while True:
            temp  = temp - self.position_aug_frame
            if temp < 0:
                break
            start_list.append(temp)
            end_list.append(temp+gap_start_end)

        temp = copy.deepcopy(tmp_start)
        while True:
            temp  = temp + self.position_aug_frame
            if (temp+gap_start_end) > self.full_size:
                break
            start_list.append(temp)
            end_list.append(temp+gap_start_end)

        data_dict['start'] = start_list
        data_dict['end'] = end_list

        return data_dict


    def new_start_end_index(self, data_dict, temp_class):

        start_idx = data_dict['start']
        end_idx = data_dict['end']

        tmp_start = start_idx - temp_class.frame_size
        tmp_end = end_idx + temp_class.frame_size

        if tmp_start < 0:
            data_dict['start'] = 0

        if tmp_end > self.full_size:
            data_dict['end'] = self.full_size

        return data_dict


    def read_data_list(self, input_list):

        new_list = list()
        temp_class = Preprocessing_data()

        for one_data in input_list:
            # data, label, mid_val
            # print(type(one_data))
            try:
                tmp_data = one_data['data']
            except IndexError:
                print(one_data)
                print(len(one_data))
                continue
            
            # return data = data, start = list, end = list
            return_dict = temp_class.cut_one_wav_file(tmp_data)
            return_dict = self.new_start_end_index(return_dict, temp_class)
            return_dict = self.make_aug_list(return_dict)
            gen_aug_data = self.gen_aug_data(return_dict, one_data)
            new_list.extend(gen_aug_data)

        return new_list









#%%
if __name__ == "__main__":
    print("hello, world~!!")
    main()


