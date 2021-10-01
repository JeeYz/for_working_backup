"""
GEN train data Ver.1.0.1

"""


import os
from scipy.io import wavfile
import numpy as np
import json
import librosa

import time
import copy
import matplotlib.pyplot as plt


FULL_SIZE = 20000

PRINT_PLT_BOOL = False
TIME_STRETCH_NUM = 2
X_AXIS_SIZE = 400
GLOBAL_THRESHOLD_RATE = 0.07
GLOBAL_NOISE_THRESHOLD_RATE = 0.001

INCLUDE_NONE_FOR_AUG = False
DO_NOT_INCLUDE_NONE = False

POP_DATA_NUM = 0
WRITE_OR_NOT = False

PREPRO_SHIFT_SIZE = 400
PREPRO_FRAME_SIZE = 1000

HEAD_SIZE = 2000
TAIL_SIZE = 2000

NORM_STAN_PARA = 'stan'

temp_para = 0


#%%
def main():
    # gen_data = control_gen_data()
    # gen_data.generate_data()
    # gen_data.generate_test_data()

    monitor_class = monitoring_train_data()
    monitor_class.length_monitoring_wav_file("D:\\voice_data_backup\\PNC_DB_ALL")
    # monitor_class.check_data_with_plt_for_npz('D:\\GEN_train_data_Ver.1.0_test_.npz')
    # monitor_class.check_data_with_plt_for_npz('D:\\test_data_20000_.npz')

    return



class monitoring_train_data():
    def __init__(self):

        self.frame_size = PREPRO_FRAME_SIZE


    def check_return_labels_rate(self, labels_dict):



        return


    def check_data_with_plt_for_npz(self, file_path):

        loaded_data = np.load(file_path)
        whole_data = loaded_data['data']
        whole_label = loaded_data['label']
        whole_filename = loaded_data['filename']

        # print(whole_label)
        # print(whole_filename)

        for_para_data = list(zip(whole_data, whole_label, whole_filename))
        # for_para_data = list(zip(whole_data, whole_label))
        # print(for_para_data)

        print("number of whole data : {len}".format(len=len(whole_data)))

        for one_data in for_para_data:
            # print(one_data)
            if one_data[1] != 5:
                print(one_data[1], one_data[2])
                # print(one_data[1])
                # self.draw_simple_graph(one_data[0])

        return


    def draw_simple_graph(self, input_data):

        fig = plt.figure()
        plt.plot(input_data)
        plt.show()

        return


    def check_train_data_correct(self):



        return


    def length_monitoring_wav_file(self, file_path):

        temp_class = Preprocessing_data()
        wav_files_list = temp_class.find_train_data_files(file_path, ".wav")

        result_list = temp_class.select_data_for_saturation(wav_files_list)
        train_label_dict = temp_class.make_data_dict(result_list)

        len_result_list = list()

        max_val = 0
        max_file = None
        min_val = 10000000
        min_file = None

        for one_key in train_label_dict.keys():
            tmp_data = train_label_dict[one_key]
            result_dict = temp_class.cut_one_wav_file(tmp_data)

            start_idx = result_dict['start']
            end_idx = result_dict['end']

            gap_size = end_idx-start_idx
            len_result_list.append(gap_size)

            if gap_size > max_val:
                max_val = gap_size
                max_file = one_key
            
            if gap_size < min_val:
                min_val = gap_size
                min_file = one_key
        
        mean_val = np.mean(len_result_list)
        max_val = np.max(len_result_list)
        min_val = np.min(len_result_list)

        print("maximum value : {max}, file : {maxfile}".format(max=max_val, maxfile=max_file))
        print("minimum value : {min}, file : {minfile}".format(min=min_val, minfile=min_file))
        print("mean value : {mean}".format(mean=mean_val))

        try:
            fig1 = plt.figure()
            plt.plot(train_label_dict[max_file])
            # plt.show()
            
            fig2 = plt.figure()
            plt.plot(train_label_dict[min_file])
            # plt.show()

            fig3 = plt.figure()
            temp_data = train_label_dict[max_file]
            temp_dict = temp_class.cut_one_wav_file(temp_data)
            plt.plot(temp_dict['data'][temp_dict['start']:temp_dict['end']])
            # plt.show()

            fig4 = plt.figure()
            temp_data = train_label_dict[min_file]
            temp_dict = temp_class.cut_one_wav_file(temp_data)
            plt.plot(temp_dict['data'][temp_dict['start']:temp_dict['end']])
            plt.show()

        except KeyError as e:
            print("error is occured...")
            print(e)

        return




#%%
class control_gen_data():
    def __init__(self):
        self.full_size = FULL_SIZE
        self.random_rate = 0.1
        # self.gen_data = list()
        # self.gen_test_data = list()

        self.label_dict = {
                'camera' : 0,
                'picture' : 1,
                'record' : 2,
                'stop' : 3,
                'end' : 4,
                'none' : 5,
            }

        self.numpy_file_path = 'D:\\GEN_train_data_Ver.1.0.npz'
        self.numpy_test_file_path = 'D:\\GEN_train_data_Ver.1.0_test_.npz'

        self.none_data_path = "D:\\voice_data_backup\\zeroth_none_label"
        self.train_data_path = "D:\\voice_data_backup\\PNC_DB_ALL"
        self.test_data_path = "D:\\voice_data_backup\\test"
        

    def generate_data(self):
        prepro = Preprocessing_data(
            none_label_path = self.none_data_path,
            train_data_path = self.train_data_path,
        )
        result_data, none_label_data = prepro.preprocess_data()
        print(len(result_data))

        if INCLUDE_NONE_FOR_AUG == True:
            result_data.extend(none_label_data)

        aug_data_pro = Augment_data()
        gen_data = aug_data_pro.read_data_list(result_data)
        print(len(gen_data))

        if PRINT_PLT_BOOL is True:
            for one in gen_data:
                if one['label'] == 'none':
                    print(json.dumps(one, sort_keys=False, indent=4, default=str, ensure_ascii=False))
                    self.print_plt_data(one['data'])

        self.print_labels_for_alldata(gen_data)
        gen_data = self.check_mod_labels_for_alldata(gen_data)
        self.print_labels_for_alldata(gen_data)

        gen_data = self.check_data_length(gen_data)

        if INCLUDE_NONE_FOR_AUG == False:
            none_label_data = self.check_mod_labels_for_alldata(none_label_data)
            none_label_data = self.check_data_length(none_label_data)
            gen_data.extend(none_label_data) 

        # print(gen_data)

        if WRITE_OR_NOT is True:
            if NORM_STAN_PARA=='stan':
                gen_data = self.standardize_data(gen_data)
            elif NORM_STAN_PARA=='norm':
                gen_data = self.normalize_data(gen_data)

            self.write_numpy_file(gen_data, self.numpy_file_path)


    def print_plt_data(self, input_data):
        fig = plt.figure()
        plt.plot(input_data)
        plt.show()   


    def write_numpy_file(self, gen_data, target_path):

        data_list = list()
        label_list = list()
        filename_list = list()

        for one_dict in gen_data:
            data_list.append(one_dict['data'])
            label_list.append(one_dict['label'])
            filename_list.append(one_dict['filename'])

        # data_list = np.array(data_list, dtype=np.float32)
        label_list = np.array(label_list, dtype=np.int16)

        np.savez(target_path, data=data_list, 
                    label = label_list, filename=filename_list
                )


    def check_data_length(self, gen_data):

        for one_dict in gen_data:
            if len(one_dict['data']) != self.full_size:
                self.print_data_with_json_dump(one_dict)
                time.sleep(10000)

        return gen_data


    def check_mod_labels_for_alldata(self, gen_data):

        for one_dict in gen_data:
            temp = one_dict['label']
            if temp not in self.label_dict.keys():
                one_dict['label'] = 'none'

        for one_dict in gen_data:
            temp = one_dict['label']
            one_dict['label'] = self.label_dict[temp]

        return gen_data


    def print_labels_for_alldata(self, gen_data):

        labels_list = list()

        for one_dict in gen_data:
            temp = one_dict['label']

            if temp not in labels_list:
                labels_list.append(temp)

        print(labels_list)


    def print_data_with_json_dump(self, tmp_dict):
        print(json.dumps(tmp_dict, sort_keys=False, indent=4, default=str, ensure_ascii=False))


    def generate_test_data(self):

        prepro = Preprocessing_data(
            test_data_path = self.test_data_path,
        )
        result_data = prepro.preprocess_test_data()
        print(len(result_data))
        # self.print_data_with_json_dump(result_data)

        self.print_labels_for_alldata(result_data)
        result_data = self.check_mod_labels_for_alldata(result_data)

        # self.print_data_with_json_dump(result_data)

        if PRINT_PLT_BOOL is True:
            for one in result_data:
                if one['label'] == 'camera':
                    # print(json.dumps(one, sort_keys=False, indent=4, default=str, ensure_ascii=False))
                    self.print_plt_data(one['data'])

        self.print_labels_for_alldata(result_data)

        if WRITE_OR_NOT is True:
            if NORM_STAN_PARA=='stan':
                result_data = self.standardize_data(result_data)
            elif NORM_STAN_PARA=='norm':
                result_data = self.normalize_data(result_data)

            self.write_numpy_file(result_data, self.numpy_test_file_path)

        return

    
    def standardize_data(self, gen_data):
        for one_dict in gen_data:
            tmp_data = one_dict['data']
            one_dict['data'] = (tmp_data-np.mean(tmp_data))/np.std(tmp_data)
        return gen_data

    
    def normalize_data(self, gen_data):
        for one_dict in gen_data:
            tmp_data = one_dict['data']
            min_val = np.min(tmp_data)
            max_val = np.max(tmp_data)
            one_dict['data'] = ((tmp_data-min_val)/(max_val-min_val))*2.0-1.0
        return gen_data



#%%
class Preprocessing_data():
    def __init__(self, **kwargs):

        if "none_label_path" in kwargs.keys():
            self.none_label_path = kwargs['none_label_path']
        else:
            self.none_label_path = None

        if "train_data_path" in kwargs.keys():
            self.train_data_path = kwargs['train_data_path']
        else:
            self.train_data_path = None

        if "test_data_path" in kwargs.keys():
            self.test_data_path = kwargs['test_data_path']
        else:
            self.test_data_path = None

        self.full_size = FULL_SIZE
        self.threshold_rate = GLOBAL_THRESHOLD_RATE
        self.noise_threshold_rate = GLOBAL_NOISE_THRESHOLD_RATE

        self.frame_size = PREPRO_FRAME_SIZE
        self.shift_size = PREPRO_SHIFT_SIZE


    def preprocess_test_data(self):
        files_list = self.find_train_data_files(
                self.test_data_path,
                ".wav"
            )

        test_label_data_dict = self.make_data_dict(files_list)
        result_test_data_list = self.cut_wav_files(test_label_data_dict, "test")

        return result_test_data_list


    def preprocess_data(self):
        
        result_list = self.find_train_data_files(
            self.none_label_path,
            ".wav"
            )
        none_data_dict = self.make_data_dict(result_list)
        
        # print_contents_of_dict(none_data_dict)
        # get_max_value(none_data_dict)

        none_label_data = self.gen_none_label_data(none_data_dict)

        print(len(none_label_data))

        files_list = self.find_train_data_files(
                self.train_data_path,
                ".wav"
            )

        # print(len(files_list))

        result_data_list = self.select_data_for_saturation(files_list)

        # print(len(result_data_list))

        train_label_data_dict = self.make_data_dict(result_data_list)
        result_train_data_list = self.cut_wav_files(train_label_data_dict, "train")

        print(len(result_train_data_list))
        print('complete~!!')

        return result_train_data_list, none_label_data


    def add_head_and_tail_none(self, input_data):

        tmp_front_num = (self.full_size - len(input_data))//2
        tmp_tail_num = self.full_size - len(input_data) - tmp_front_num

        # low_val = np.min(input_data)
        # high_val = np.max(input_data)
        
        # front_random_data = np.random.randint(low_val, high_val, tmp_front_num)
        # tail_random_data = np.random.randint(low_val, high_val, tmp_tail_num)

        # front_random_data = front_random_data*self.noise_threshold_rate
        # tail_random_data = tail_random_data*self.noise_threshold_rate

        front_random_data = np.zeros(tmp_front_num, dtype=np.float32)
        tail_random_data = np.zeros(tmp_tail_num, dtype=np.float32)

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

                temp_dict = dict()
                temp_dict['data'] = tmp
                temp_dict['label'] = 'none'
                temp_dict['mid_val'] = -1
                temp_dict['filename'] = one_key

                if len(tmp) != self.full_size:
                    print("occured exception...")
                    continue

                none_data_list.append(temp_dict)

                tmp_num = -1*self.full_size
                tmp = tmp_data[tmp_num:]

            elif tmp_range >= tmp_len\
                        and\
                    tmp_len > self.full_size:
                tmp = tmp_data[:self.full_size]

            else:
                tmp = self.add_head_and_tail_none(tmp_data)
            
            temp_dict = dict()
            temp_dict['data'] = tmp
            temp_dict['label'] = 'none'
            temp_dict['mid_val'] = -1
            temp_dict['filename'] = one_key

            if len(tmp) != self.full_size:
                print("occured exception...")
                continue

            none_data_list.append(temp_dict)
            

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

            w_data = np.array(w_data, dtype=np.float32)
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
        temp_dict['filename'] = filename

        return temp_dict


    def make_frame_list(self, data_size):

        result_list = list()

        tmp = data_size//self.shift_size

        for i in range(tmp):
            tmp_start = i*self.shift_size
            if (tmp_start+self.frame_size)>data_size:
                break

            result_list.append(tmp_start)

        return result_list


    def gen_mean_list(self, curr_data, start_list):
        
        mean_list = list()

        for i in start_list:
            tmp_end = i+self.frame_size
            tmp_data = curr_data[i:tmp_end]
            tmp_data = np.abs(tmp_data)
            tmp_mean = np.mean(tmp_data)
            mean_list.append(tmp_mean)

        return mean_list


    def cut_one_wav_file(self, input_data):

        frame_start_list = self.make_frame_list(len(input_data))
        mean_list = self.gen_mean_list(input_data, frame_start_list)

        max_mag = np.max(mean_list)
        threshold_val = max_mag*self.threshold_rate

        start_idx = 0
        end_idx = 0

        data_start_list = list(zip(frame_start_list, mean_list))

        for ds in data_start_list:
            if threshold_val < ds[1]:
                start_idx = ds[0]
                break

        for ds in reversed(data_start_list):
            if threshold_val < ds[1]:
                end_idx = ds[0]+self.frame_size
                break

        one_data_dict = dict()

        if end_idx == 0:
            end_idx = len(input_data)

        one_data_dict['data'] = input_data
        one_data_dict['start'] = start_idx
        one_data_dict['end'] = end_idx

        return one_data_dict


    # fit full size
    def modifying_data(self, input_dict):

        tmp_start = input_dict['start']
        tmp_end = input_dict['end']
        tmp_data = input_dict['data']

        low_val = np.min(tmp_data)
        high_val = np.max(tmp_data)

        half_fullsize = self.full_size//2
        mid_point = (tmp_end+tmp_start)//2 + half_fullsize
        # print(len(tmp_data), mid_point, tmp_start, tmp_end)

        # tmp_fdata = np.random.randint(low_val, high_val, half_fullsize)
        # tmp_edata = np.random.randint(low_val, high_val, half_fullsize)

        # tmp_fdata = tmp_fdata * self.noise_threshold_rate
        # tmp_edata = tmp_edata * self.noise_threshold_rate

        tmp_fdata = np.zeros(half_fullsize, dtype=np.float32)
        tmp_edata = np.zeros(half_fullsize, dtype=np.float32)

        tmp_data = np.append(tmp_fdata, tmp_data)
        tmp_data = np.append(tmp_data, tmp_edata)

        new_start = mid_point - half_fullsize
        new_end = mid_point + half_fullsize
        
        tmp_main_data = tmp_data[new_start:new_end]

        if len(tmp_main_data) != self.full_size:
            print("this data is incorrent size...")
            print(new_start, new_end)
            print(len(tmp_main_data), len(tmp_data))
            return "None", "None"
        else:
            return tmp_main_data, (mid_point-new_start)
      

    def mod_test_data(self, input_dict):
        tmp_start = input_dict['start']
        tmp_end = input_dict['end']
        tmp_data = input_dict['data']

        half_fullsize = self.full_size//2
        mid_point = (tmp_end+tmp_start)//2 + half_fullsize
        # print(len(tmp_data), mid_point, tmp_start, tmp_end)

        tmp_fdata = np.zeros(half_fullsize, dtype=np.float32)
        tmp_edata = np.zeros(half_fullsize, dtype=np.float32)

        tmp_data = np.append(tmp_fdata, tmp_data)
        tmp_data = np.append(tmp_data, tmp_edata)

        new_start = mid_point - half_fullsize
        new_end = mid_point + half_fullsize
        
        tmp_main_data = tmp_data[new_start:new_end]

        if len(tmp_main_data) != self.full_size:
            print("this data is incorrent size...")
            print(new_start, new_end)
            print(len(tmp_main_data), len(tmp_data))
            return "None", "None"
        else:
            return tmp_main_data, (mid_point-new_start)


    def cut_wav_files(self, data_dict, data_flag):
        
        global temp_para

        result_data = list()

        for one_key in data_dict.keys():

            tmp_data = data_dict[one_key]
            
            output_data_dict = self.cut_one_wav_file(tmp_data)
            
            # if output_data_dict['start'] < HEAD_SIZE:
            #     print(output_data_dict['start'], one_key)
            #     temp_para+=1

            # fit to full size
            if data_flag is "train":
                one_tmp_data, mid_point = self.modifying_data(output_data_dict)
            elif data_flag is "test":
                one_tmp_data, mid_point = self.mod_test_data(output_data_dict)


            if one_tmp_data != 'None':
                # print("funfun")
                # print(one_key)
                output_dict = self.add_label_info(
                        one_tmp_data, one_key, mid_point
                    )
                result_data.append(output_dict)
        
        print(temp_para)
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

        if TIME_STRETCH_NUM == 2:
            self.time_stretch_list = [1.0, 1.1,]
        elif TIME_STRETCH_NUM == 1:
            self.time_stretch_list = [1.0,]

        self.position_aug_frame = X_AXIS_SIZE

        self.threshold_rate = GLOBAL_THRESHOLD_RATE
        self.noise_threshold_rate = GLOBAL_NOISE_THRESHOLD_RATE

        self.para_num = 0


    def add_noise_data(self, input_data, asize):

        low_val = np.min(input_data)
        high_val = np.max(input_data)

        # front_random_data = np.random.randint(low_val, high_val, asize)
        # front_random_data = front_random_data*self.noise_threshold_rate

        front_random_data = np.zeros(asize, dtype=np.float32)

        return front_random_data


    def fit_full_size(self, input_data):

        low_val = np.min(input_data)
        high_val = np.max(input_data)

        if len(input_data) > self.full_size:
            temp_num = len(input_data) - self.full_size
            temp_num_1 = temp_num//2
            temp_num_2 = temp_num - temp_num_1

            temp_data = input_data[temp_num_1:]
            temp_num_2 = -1*temp_num_2
            temp_data = temp_data[:temp_num_2]

            if len(temp_data) != self.full_size:
                print(len(temp_data))
                time.sleep(10000)
            
        else:
            temp_num = self.full_size - len(input_data)
            temp_num_1 = temp_num//2
            temp_num_2 = temp_num - temp_num_1

            # temp_random_data_1 = np.random.randint(low_val, high_val, temp_num_1)    
            # temp_random_data_1 = temp_random_data_1*self.noise_threshold_rate
            # temp_random_data_2 = np.random.randint(low_val, high_val, temp_num_2)    
            # temp_random_data_2 = temp_random_data_2*self.noise_threshold_rate

            temp_random_data_1 = np.zeros(temp_num_1, dtype=np.float32)    
            temp_random_data_2 = np.zeros(temp_num_2, dtype=np.float32)    

            temp_data = np.append(temp_random_data_1, input_data)
            temp_data = np.append(temp_data, temp_random_data_2)

            if len(temp_data) != self.full_size:
                print(len(temp_data))
                time.sleep(10000)

        return temp_data


    def gen_aug_data(self, data_dict, one_data):

        new_data_list = list()

        init_start_idx = data_dict['start'][0]
        init_data = copy.deepcopy(data_dict['data'])
        init_label = one_data['label']
        init_filename = one_data['filename']

        # print(data_dict['start'])

        for one_st in data_dict['start']:
            temp = dict()

            cond1 = one_st - init_start_idx
            add_size = copy.deepcopy(cond1)

            if add_size != 0:
                add_size = np.abs(add_size)
                return_noise = self.add_noise_data(init_data, add_size)
                if cond1 > 0:
                    # print(len(aug_data), len(return_noise))
                    temp_data = np.append(return_noise, init_data)
                    # temp_data = np.array(temp_data, dtype=np.float32)
                    new_data = temp_data[0: self.full_size]
                    # print(len(temp_data), len(new_data))
                else:
                    # print(len(aug_data), len(return_noise))
                    temp_data = np.append(init_data, return_noise)
                    # temp_data = np.array(temp_data, dtype=np.float32)
                    t_size = -1*self.full_size
                    new_data = temp_data[t_size:]
                    # print(len(temp_data), len(new_data))
                
            else:
                new_data = copy.deepcopy(init_data)


            if len(new_data) != self.full_size:
                new_data = self.fit_full_size(new_data)

            # new_data = np.array(new_data, dtype=np.float32)

            temp['data'] = new_data
            temp['label'] = init_label
            temp['filename'] = init_filename

            new_data_list.append(temp)

            self.para_num += 1
            # if self.para_num%10 == 0:
            # print("{} th data generated...".format(self.para_num), end='\r')

        return new_data_list


    def make_aug_list(self, data_dict):

        start_list = list()
        end_list = list()

        tmp_start = data_dict['start']
        tmp_end = data_dict['end']

        # print("before start of data dict : {start}".format(start=data_dict['start']))

        start_list.append(tmp_start)
        end_list.append(tmp_end)

        gap_start_end = tmp_end - tmp_start
        if tmp_end == 0:
            print(tmp_start, tmp_end, gap_start_end)
            print(data_dict['data'])
            fig = plt.figure()
            plt.plot(data_dict['data'])
            plt.show()

        temp = copy.deepcopy(tmp_start)
        while True:
            temp  = temp - self.position_aug_frame
            if temp < 0:
                break
            start_list.append(temp)
            end_list.append(temp+gap_start_end)

        for i in range(POP_DATA_NUM):
            if len(start_list) > 1:
                start_list.pop()
            if len(end_list) > 1:
                end_list.pop()

        temp = copy.deepcopy(tmp_start)
        while True:
            temp  = temp + self.position_aug_frame
            if (temp+gap_start_end) > self.full_size:
                break
            start_list.append(temp)
            end_list.append(temp+gap_start_end)

        for i in range(POP_DATA_NUM):
            if len(start_list) > 1:
                start_list.pop()
            if len(end_list) > 1:
                end_list.pop()

        data_dict['start'] = start_list
        data_dict['end'] = end_list

        # print("after start of data dict : {start}".format(start=data_dict['start']))

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
                init_data = one_data['data']
            except IndexError:
                print(one_data)
                print(len(one_data))
                print(IndexError)
                time.sleep(10000)
            
            for r in self.time_stretch_list:
                tmp_data = copy.deepcopy(init_data)
                # print(type(tmp_data))
                # print(tmp_data)
                # tmp_data = np.array(tmp_data, dtype=np.float32)
                aug_data = librosa.effects.time_stretch(tmp_data, r)
                
                return_dict = temp_class.cut_one_wav_file(aug_data)

                return_dict = self.new_start_end_index(return_dict, temp_class)
                return_dict = self.make_aug_list(return_dict)

                print(return_dict['start'])
                print(return_dict['end'])
                print('\n')

                gen_aug_data = self.gen_aug_data(return_dict, one_data)
                new_list.extend(gen_aug_data)

        print("total data num : {len}".format(len=len(new_list)))

        return new_list









#%%
if __name__ == "__main__":
    print("hello, world~!!")
    main()


