import numpy as np
import json
import os

from scipy.io import wavfile
import librosa

import matplotlib.pyplot as plt

sr = 16000 # sample rate

tail_size = 3000

# full_size = 20000
full_size = 32000
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





def refine_train_data_files(input_list):

    command_list = ["picture", "stop", "record"]

    result_list = list()

    for one_file in input_list:
        temp_line = one_file.split("\\")
        if temp_line[-2] in command_list:
            result_list.append(one_file)

    return result_list





import numpy as np
import matplotlib.pyplot as plt


train_npz_file_path = "D:\\train_data_mini_20000_random_.npz"
# train_npz_file_path = "D:\\train_data_test_20000_random_.npz"
test_npz_file_path = "D:\\test_data_20000_.npz"




def draw_graphes(new_files_list, new_test_files):
    
    plt_graph_save_path = "D:\\temp_graphes_images\\"
    plt_graph_filename = "selected_train_and_test_data_"

    col_num = 7
    row_num = 7

    file_num = 0
    data_num = 0

    cond1 = len(new_files_list)
    while data_num < cond1:

        fig, axs = plt.subplots(    
                col_num, row_num,
                figsize=(12, 12), 
                constrained_layout=True
                                )

        for n in range(col_num):
            for m in range(row_num):
                if data_num >= cond1:
                    break

                one_file = new_files_list[data_num]
                sample_rate, data = wavfile.read(one_file)
                data = np.array(data, dtype=np.float32)
                data = fit_determined_size(data)

                temp_text = one_file.split("\\")[-2]

                axs[n, m].plot(data)
                axs[n, m].set_title(temp_text)
                data_num += 1
                print("{}th file is done...{}".format(file_num+1, data_num), end='\r')

                if data_num >= cond1:
                    break

        filename_str = plt_graph_save_path + plt_graph_filename + str('%02d'%file_num) + ".png"
        plt.savefig(filename_str, dpi=300)
        file_num += 1

    
    data_num = 0

    cond2 = len(new_test_files)
    while data_num < cond2:

        fig, axs = plt.subplots(    col_num, row_num,
                                    figsize=(10, 10), 
                                    constrained_layout=True
                                )

        for n in range(col_num):
            for m in range(row_num):
                # print(len(cl_test_data), data_num)
                if data_num >= cond2:
                    break
                
                one_file = new_test_files[data_num]
                sample_rate, data = wavfile.read(one_file)
                data = np.array(data, dtype=np.float32)
                data = fit_determined_size(data)

                temp_text = one_file.split("\\")[-2]

                axs[n, m].plot(data)
                axs[n, m].set_title(temp_text)
                data_num += 1
                if data_num >= cond2:
                    break

        filename_str = plt_graph_save_path + plt_graph_filename + str('%02d'%file_num) + ".png"
        plt.savefig(filename_str, dpi=300)
        file_num += 1


    return




def main():

    train_data_files = search_all_satu_files()

    print("generate train data...")

    new_files_list = refine_train_data_files(train_data_files)

    print("refine complete~!!")

    print(len(new_files_list))

    test_files_list = find_files(   
            file_ext=".wav",
            filepath="D:\\voice_data_backup\\test\\"
                                )

    new_test_files = refine_train_data_files(test_files_list)

    print("refine complete~!!")

    print(len(new_test_files))

    draw_graphes(new_files_list, new_test_files)


    return










if __name__ == '__main__':
    print("hello, world~!!")
    main()
