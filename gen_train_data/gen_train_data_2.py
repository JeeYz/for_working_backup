from pickle import STACK_GLOBAL
import numpy as np
import json
import os
from numpy.core.shape_base import stack

from scipy.io import wavfile
import librosa

import time
import random

sr = 16000 # sample rate

tail_size = 3000

full_size = 20000
trigger_val = 1.0

slice_data_num = 10000

file_ext = ".wav"
parent_folder = "D:\\voice_data_backup\\PNC_DB_ALL"

command_dict = {    
                "camera": 0,
                "picture": 1,
                "record": 2, 
                "stop": 3,
                "end": 4,                
                }

rate_list = [
                1.10, 
                1.0,
                # 0.9,
            ]

# random_value = [
#                     # 3000, 3500, 4000, 4500,
#                     # 5000, 5500, 6000, 6500
#                     2000, 3000, 4000,
#                     5000, 6000,
#                 ]

none_label_num = 5


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ## function : cut_input_signal_v2
# about :   입력되는 신호를 신호의 세기를 기준으로 잘라 주는 함수
#           앞 부분만 자르고 뒤는 그대로 두는 함수
# input : signal data, frame size, shift size
# output : 잘린 데이터
def cut_input_signal_v2(data, **kwargs):

    if "frame_size" in kwargs.keys():
        frame_size = kwargs['frame_size']
    if "shift_size" in kwargs.keys():
        shift_size = kwargs['shift_size']

    gap_frame_shift = frame_size-shift_size

    num_frames = len(data)//(gap_frame_shift)

    mean_val_list = list()

    for i in range(num_frames):
        temp_n = i*gap_frame_shift
        one_frame_data = data[temp_n:temp_n+frame_size]
        mean_val_list.append(np.mean(np.abs(one_frame_data)))

    for i,start in enumerate(mean_val_list):
        if trigger_val < start:
            start_index = i
            break
        else:
            start_index = 0

    front_size = np.random.randint(40, 70)*100
    
    temp = gap_frame_shift*start_index-front_size

    if temp < 0:
        abs_temp = np.abs(temp)
        result = np.append(np.zeros(abs_temp), data)
        # result = data
    elif temp == 0:
        result = data
    else:
        result = data[temp:]

    return result




# ## function : cut_input_signal_v4
# about : 입력되는 신호를 신호의 세기를 기준으로 잘라 주는 함수
# input : signal data, frame size, shift size
# output : 잘린 데이터
def cut_input_signal_v4(data, random_value, **kwargs):

    if "frame_size" in kwargs.keys():
        frame_size = kwargs['frame_size']
    if "shift_size" in kwargs.keys():
        shift_size = kwargs['shift_size']

    gap_frame_shift = frame_size-shift_size

    num_frames = len(data)//(gap_frame_shift)

    mean_val_list = list()

    for i in range(num_frames):
        temp_n = i*gap_frame_shift
        one_frame_data = data[temp_n:temp_n+frame_size]
        mean_val_list.append(np.mean(np.abs(one_frame_data)))

    for i,start in enumerate(mean_val_list):
        if trigger_val < start:
            start_index = i
            break
        else:
            start_index = 0

    for i,end in enumerate(reversed(mean_val_list)):
        if trigger_val < end:
            end_index = len(mean_val_list)-i-1
            break
        else:
            end_index = len(mean_val_list)

    front_size = random_value
    temp = gap_frame_shift*start_index-front_size

    if temp <= 0:
        temp = 0

    temp_tail = gap_frame_shift*end_index+tail_size

    if temp_tail > len(data):
        temp_tail = len(data)-1

    result = np.append(np.zeros(temp), data[temp:temp_tail])

    return result





# ## function : cut_input_signal_v5
# about :  테스트 데이터를 위한 함수
# input : signal data, frame size, shift size
# output : 잘린 데이터
def cut_input_signal_v5(data, **kwargs):

    if "frame_size" in kwargs.keys():
        frame_size = kwargs['frame_size']
    if "shift_size" in kwargs.keys():
        shift_size = kwargs['shift_size']

    gap_frame_shift = frame_size-shift_size

    num_frames = len(data)//(gap_frame_shift)

    mean_val_list = list()

    for i in range(num_frames):
        temp_n = i*gap_frame_shift
        one_frame_data = data[temp_n:temp_n+frame_size]
        mean_val_list.append(np.mean(np.abs(one_frame_data)))

    for i,start in enumerate(mean_val_list):
        if trigger_val < start:
            start_index = i
            break
        else:
            start_index = 0

    front_size = 5000
    
    temp = gap_frame_shift*start_index-front_size

    if temp < 0:
        abs_temp = np.abs(temp)
        result = np.append(np.zeros(abs_temp), data)
        # result = data
    elif temp == 0:
        result = data
    else:
        result = data[temp:]

    return result




# ## function : cut_input_signal_v6
# about :   입력되는 신호를 신호의 세기를 기준으로 잘라 주는 함수
#           앞 부분만 자르고 뒤는 그대로 두는 함수
# input : signal data, frame size, shift size, random value
# output : 잘린 데이터
def cut_input_signal_v6(data, random_val, **kwargs):

    if "frame_size" in kwargs.keys():
        frame_size = kwargs['frame_size']
    if "shift_size" in kwargs.keys():
        shift_size = kwargs['shift_size']

    gap_frame_shift = frame_size-shift_size

    num_frames = len(data)//(gap_frame_shift)

    mean_val_list = list()

    for i in range(num_frames):
        temp_n = i*gap_frame_shift
        one_frame_data = data[temp_n:temp_n+frame_size]
        mean_val_list.append(np.mean(np.abs(one_frame_data)))

    for i,start in enumerate(mean_val_list):
        if trigger_val < start:
            start_index = i
            break
        else:
            start_index = 0

    front_size = random_val
    
    temp = gap_frame_shift*start_index-front_size

    if temp < 0:
        abs_temp = np.abs(temp)
        result = np.append(np.zeros(abs_temp), data)
        # result = data
    elif temp == 0:
        result = data
    else:
        result = data[temp:]

    return result




# ## function : fit_determined_size
# about : 잘려진 데이터의 길이를 정해진 길이에 맞춰 주는 함수
# input : 잘린 신호 데이터
# output : 정해진 길이로 맞춰진 데이터
def fit_determined_size(data, **kwargs):
    # print(full_size-len(data))
    if len(data) > full_size:
        return data[0:full_size]
    elif len(data) == full_size:
        return data
    else:
        return np.append(data, np.zeros(full_size-len(data)))




def standardize_signal(data, **kwargs):
    return (data-np.mean(data))/np.std(data)




# ## function : write_numpy_file
# about : 데이터를 numpy 파일로 출력
# input : 데이터 (list)
# output : None
def write_numpy_file(full_data, **kwargs):

    if "slice_data" in kwargs.keys():
        slice_data = kwargs['slice_data']
    else:
        slice_data = None

    if 'zeroth_bool' in kwargs.keys():
        zeroth_bool = kwargs['zeroth_bool']
    else:
        zeroth_bool = False

    if 'label_numpy' in kwargs.keys():
        label_numpy = kwargs['label_numpy']

    if 'default_filename' in kwargs.keys():
        file_name = kwargs['default_filename']
    else:
        file_name = './train_data_'



    if slice_data == None:
        if zeroth_bool == True:
            label_np = np.array(['None' for i in range(len(full_data))])
        else:
            label_np = label_numpy

        full_data = np.array(full_data)
        
        np.savez(file_name, data=full_data, label=label_np)

    else:
        temp_list = list()
        file_num = 0
        default_path = file_name

        for i, one in enumerate(full_data):
            # print(slice_data)
            if i%slice_data==0 and i != 0:
                file_path = default_path+str('%03d'%file_num)+'.npz'

                if zeroth_bool == True:
                    label_np = np.array(['None' for i in range(len(temp_list))])
                else:
                    label_np = label_numpy
                
                # print(len(temp_list), i)
                data_np = np.array(temp_list)
                np.savez(file_path, data=data_np, label=label_np)
                temp_list = list()
                temp_list.append(one)
                file_num += 1
            else:
                temp_list.append(one)
        
        file_num += 1
        file_path = default_path+str('%03d'%file_num)+'.npz'
        label_np = np.array(['None' for i in range(len(temp_list))])
        data_np = np.array(temp_list)
        np.savez(file_path, data=temp_list, label=label_np)

    return



# ## function : make_label_list
# about : 파일 리스트에 있는 경로명을 보고 label 리스트를 작성
# input : 파일 리스트 (list)
# output : 레이블 데이터 (list)
def make_label_list(files_list, random_value, **kwargs):
    if "kind_of_data" in kwargs.keys():
        kind_of_data = kwargs["kind_of_data"]

    label_result = list()
    label_dict = {  'camera' : 0,
                    'picture' : 1,
                    'record' : 2,
                    'stop' : 3,
                    'end' : 4}

    if kind_of_data == "train":

        for one in files_list:
            parsed_path = one.split('\\')
            # print(parsed_path[-2], label_dict.keys())
            if parsed_path[-2] in label_dict.keys():
                for r in rate_list:
                    for rv in random_value:
                        label_result.append(label_dict[parsed_path[-2]])
                    # label_result.append(label_dict[parsed_path[-2]])
            else:
                for r in rate_list:
                    for rv in random_value:
                        label_result.append(5)
                    # label_result.append(5)

        return label_result
    
    elif kind_of_data == "test":

        for one in files_list:
            parsed_path = one.split('\\')
            if parsed_path[-2] in label_dict:
                label_result.append(label_dict[parsed_path[-2]])
            else:
                label_result.append(5)

        return label_result





'''
@ func : 데이터 증강 함수
@ param : raw data
@ return : 증강된 데이터 list
@ remark : for문을 핵심으로 개발
@ data : 
'''
def augment_data(data):

    result = list()
    for r in rate_list:
        aug_data = librosa.effects.time_stretch(data, r)

        if len(aug_data) > full_size:
            result.append(aug_data[:full_size])
        else:
            result.append(aug_data)

    return result




# ## function : write_numpy_file
# about : 해당 경로의 wav 파일을 읽어서 int형 데이터 반환
# input : 파일 경로
# output : 해당 파일 데이터
def read_wav_file(**kwargs):
    if 'file_path' in kwargs.keys():
        file_path = kwargs['file_path']
    _, data = wavfile.read(file_path)
    return data




# ## function : find_files
# about : 파일을 탐색해 주는 함수
# input : 타겟 경로와 확장자
# output : 파일 리스트 
def find_files(**kwargs):
    if "filepath" in kwargs.keys():
        filepath = kwargs['filepath']
    if "file_ext" in kwargs.keys():
        file_ext = kwargs['file_ext']

    files_list = list()
    for (path, dir, files) in os.walk(filepath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                # print("%s\\%s" % (path, filename))
                one_file_name = path+'\\'+filename
                files_list.append(one_file_name)

    return files_list




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++












def determine_condition(file_name):

    try:
        sample_rate, data = wavfile.read(file_name)
    except wavfile.WavFileWarning:
        print("\noccur warning...\n")

    max_value = np.max(data)
    min_value = np.min(data)
    sorted_np, indices = np.unique(data, return_counts=True)

    # 조건 0 and 1
    if max_value != 32767   \
                and         \
            min_value != -32768:                        

        return 1
    else:
        return 0




def search_all_satu_files():

    all_norm_data = list()

    for (path, dir, files) in os.walk(parent_folder):   
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                file_name = path + "\\" + filename
                cond = determine_condition(file_name)
                if cond == 1:
                    all_norm_data.append(file_name)

    return all_norm_data




def make_all_files_list():

    all_files = list()
    parent_folder = "D:\\voice_data_backup\\PNC_DB_ALL"
    file_ext = '.wav'

    for (path, dir, files) in os.walk(parent_folder):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                file_name = path + "\\" + filename
                all_files.append(file_name)

    return all_files




def generate_train_data(train_data_files, label_list):

    train_data = list()

    for one in train_data_files:
        sr, data = wavfile.read(one)
        # print(len(data), end='  ')
        data = np.array(data, dtype=np.float32)
        
        train_data.append(data)

    # train_data = np.array(train_data, dtype=np.float32)
    label_list = np.array(label_list, dtype=np.int16)



    npz_filepath = "D:\\train_data_or_condition.npz"

    np.savez(npz_filepath, data=train_data, label = label_list)




def check_files_path(norm_data, null_list, all_files):

    npz_file = "D:\\train_data_satu_or_condition.npz"

    train_data_path = list()
    label_list = list()

    for one_file in all_files:
        none_flag = 0
        for one_command in norm_data.keys():
            if one_file in norm_data[one_command]:
                train_data_path.append(one_file)
                label_list.append(command_dict[one_command])
                none_flag = 1
                break
                
            elif one_file in null_list:
                none_flag = 1
                break
                
        if none_flag == 0:
            train_data_path.append(one_file)
            label_list.append(none_label_num)

    print("num of path: {}, num of label: {}".format(len(train_data_path), len(label_list)))

    return train_data_path, label_list




def initialize_dict(data_dict):
    
    for one in command_dict.keys():
        data_dict[one] = list()
    
    return data_dict




def read_json_file():

    new_dict = dict()
    new_dict = initialize_dict(new_dict)

    null_list = list()

    json_file = "or_condition_data.json"
    with open(json_file, 'r', encoding='utf-8') as jf:
        json_data = json.load(jf)

    for one_speaker in json_data["speakers"]:
        for one_command in one_speaker["commands"]:
            if one_command["normal_data"] != []:
                for npath in one_command["normal_data"]:
                    new_dict[one_command["command"]].append(npath)
                
            if one_command["saturation_data"] != []:
                for npath in one_command["saturation_data"]:
                    null_list.append(npath)

    return new_dict, null_list







def devide_data(train_data_file):

    train_data_list = list()
    test_data_list = list()

    temp_command = "null"
    temp_speaker = "null"
    stack_file = list()

    for i,one_file in enumerate(train_data_file):
        print(i, '\t', one_file)

        temp_line = one_file.split("\\")
        this_command = temp_line[-2]
        this_speaker = temp_line[-3]

        if temp_command == "null":
            temp_command = this_command
            temp_speaker = this_speaker
        
        if temp_command != this_command or temp_speaker != this_speaker:

            print(one_file, temp_command, this_command, temp_speaker, this_speaker, len(stack_file))

            if len(stack_file) > 1:
                max_num = len(stack_file)-1
                temp_num = random.randint(0, max_num)
                test_data_list.append(stack_file[temp_num])
                del stack_file[temp_num]

            for one in stack_file:
                train_data_list.append(one)

            stack_file = list()
            stack_file.append(one_file)
            temp_command = this_command
            temp_speaker = this_speaker

        else:
            stack_file.append(one_file)
    
    if len(stack_file) > 1:
        max_num = len(stack_file)-1
        temp_num = random.randint(0, max_num)
        test_data_list.append(stack_file[temp_num])
        del stack_file[temp_num]

    for one in stack_file:
        train_data_list.append(one)

    stack_file = list()

    return train_data_list, test_data_list




def main():

    # norm_data, null_list = read_json_file()
    # print("read json file...")

    # all_files = make_all_files_list()
    # print("make files list...")

    # train_data_files, label_list = check_files_path(norm_data, null_list, all_files)
    # print("refine files path...")

    # generate_train_data(train_data_files, label_list)
    # print("generate all data...")


    # train data 생성

    # train_data_file_name = "D:\\train_data_mini_20000_random_"
    # train_data_file_name = "D:\\train_data_middle_20000_random_"
    train_data_file_name = "D:\\train_data_middle_20000_random_confirm_0_"
    # train_data_file_name = "D:\\train_data_test_20000_random_"



    train_data_ori = search_all_satu_files()
    train_data_files, test_data_files = devide_data(train_data_ori)

    print("generate train data...")

    train_data_set = list()
    train_label_set = list()
    label_list = list()

    random_value = list()

    random_num = 20

    while len(random_value)<random_num:
        rand_val = np.random.randint(45, 80)*100
        if rand_val not in random_value:
            random_value.append(rand_val)

    label_list = make_label_list(train_data_files, random_value, kind_of_data="train")

    file_name = './train_data_'

    file_num = 0
    data_num = 0

    label_num = 0
    count_num = 0


    print(len(train_data_files), len(label_list))
    print(len(train_data_ori), len(train_data_files), len(test_data_files))
    # time.sleep(1000)


    for i, one_file in enumerate(train_data_files):

        one_data = read_wav_file(file_path=one_file)
        one_data = np.array(one_data, dtype=np.float32)
        one_data = standardize_signal(one_data)

        for rv in random_value:

            result = cut_input_signal_v6(one_data, rv, frame_size=400, shift_size=200)

            # draw_graph(result)

            result_list = augment_data(result)

            for one_r in result_list:

                result = fit_determined_size(one_r)
                # result = add_noise_data(result)

                train_data_set.append(result)

                data_num+=1


                print("\r{}th file is done...".format(data_num), end='')
                count_num+=1

        
        # result = cut_input_signal_v2(one_data, frame_size=400, shift_size=200)

        # # draw_graph(result)

        # result_list = augment_data(result)

        # for one_r in result_list:

        #     result = fit_determined_size(one_r)
        #     # result = add_noise_data(result)

        #     train_data_set.append(result)

        #     data_num+=1


        #     print("\r{}th file is done...".format(data_num), end='')
        #     count_num+=1

    
    # write_numpy_file(train_datya_set,
    #                 label_nump=np.array(label_list))           
    
    write_numpy_file(train_data_set,
                    label_numpy=np.array(label_list),
                    default_filename=train_data_file_name)  

    # print(label_num, len(label_list[label_num:]))
    # time.sleep(1000)

    print('\n')
    print(len(label_list), count_num)





    # test data 생성

    print("hello, world~!!")
    # files_list = list()
    # files_list = find_files(filepath='D:\\Speech\\test',
    #             file_ext='.wav')
    # files_list = find_files(filepath='D:\\voice_data_backup\\test',
    #             file_ext='.wav')
    # print(files_list)

    test_data_set = list()
    label_list = list()

    # label_list = make_label_list(files_list, kind_of_data="test")
    label_list = make_label_list(test_data_files, [], kind_of_data="test")

    print('\n', len(label_list))

    # for i, one_file in enumerate(files_list):
    for i, one_file in enumerate(test_data_files):

        one_data = read_wav_file(file_path=one_file)
        one_data = standardize_signal(one_data)

        result = cut_input_signal_v5(one_data, frame_size=400, shift_size=200)

        # draw_graph(result)

        result = fit_determined_size(result)
        # print(len(result))
        # result = add_noise_data(result)

        test_data_set.append(result)

        print("\r{}th file is done...".format(i), end='')

    write_numpy_file(test_data_set,
                    label_numpy=np.array(label_list), 
                    default_filename='D:\\test_data_20000_confirm_0_')




    return










if __name__ == '__main__':
    print("hello, world~!!")
    main()
