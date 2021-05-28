# ## 필요한 기능
# 1. 신호를 걸러 주는 기능
# 2. 결과 데이터를 길이에 맞게 만들어 주는 기능
# 3. 미세한 노이즈 추가 기능
# 4. 결과 완성된 데이터 파일로 출력
# 5. 결과를 검증할 수 있는 그래프 출력
# 6. 해당 폴더의 wav 파일 탐색 기능
# 7. 종합 실행 함수
# etc. 딱 하나의 기능만을 수행하는 함수 작성
# ##

import os
import matplotlib.pyplot as plt

import numpy as np
import time

from scipy.io import wavfile


sr = 16000 # sample rate
front_size = 4000
tail_size = 4000
full_size = sr*4
trigger_val = 1.5

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


# ## function : make_train_data_file
# about : 입력되는 신호에 대해 결과 파일로 만들어 주는 종합 함수
# input : signal data
# output : None
def make_train_data_file(inputs, **kwargs):



    return


# ## function : cut_input_signal_for_zeroth
# about : zeroth 데이터로 none 레이블 생성을 위한 함수
# input : signal data, frame size, shift size
# output : 잘린 데이터
def cut_input_signal_for_zeroth(data, **kwargs):

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


    temp = gap_frame_shift*start_index-front_size

    if temp <= 0:
        temp = 0

    if len(data) > full_size:
        temp_tail = full_size-8000
    elif len(data) == full_size:
        temp_tail = full_size
    else:
        temp_tail = len(data)-8000

    result = data[temp:temp_tail]

    return result


# ## function : cut_input_signal
# about : 입력되는 신호를 신호의 세기를 기준으로 잘라 주는 함수
# input : signal data, frame size, shift size
# output : 잘린 데이터
def cut_input_signal(data, **kwargs):

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

    temp = gap_frame_shift*start_index-front_size

    if temp <= 0:
        temp = 0

    temp_tail = gap_frame_shift*end_index+tail_size

    if temp_tail > len(data):
        temp_tail = len(data)-1

    result = data[temp:temp_tail]

    return result


# ## function : fit_determined_size
# about : 잘려진 데이터의 길이를 정해진 길이에 맞춰 주는 함수
# input : 잘린 신호 데이터
# output : 정해진 길이로 맞춰진 데이터
def fit_determined_size(data, **kwargs):
    # print(full_size-len(data))
    return np.append(data, np.zeros(full_size-len(data)))


# ## function : add_noise_data
# about : 노이즈를 추가해 주는 데이터
# input : 신호 데이터
# output : 잡음이 추가된 데이터
def add_noise_data(data, **kwargs):
    
    noise_data = np.random.randn(full_size)*0.01
    
    return data+noise_data


# ## function : draw_graph
# about : 신호 데이터를 그래프로 그려주는 함수
# input : 신호 데이터
# output : None
def draw_graph(data, **kwargs):

    plt.figure()
    plt.plot(data)

    plt.xlabel('sample rate')
    plt.ylabel('amplitude')
    plt.title('A signal data')

    plt.tight_layout()
    plt.show()

    return


# ## function : standardize_signal
# about : 입력된 신호 데이터를 표준화 시켜주는 함수
# input : 신호 데이터
# output : 표준화된 데이터
def standardize_signal(data, **kwargs):
    return (data-np.mean(data))/np.std(data)


# ## function : write_file
# about : 정제된 데이터에 대해 numpy 혹은 wav 파일로 출력
# input : 데이터 (not list)
# output : None
def write_file(data, **kwargs):
    if "data_path" in kwargs.keys():
        data_path = kwargs["data_path"]

    wavfile.write(data_path, sr, data)

    return


# ## function : make_label_list
# about : 파일 리스트에 있는 경로명을 보고 label 리스트를 작성
# input : 파일 리스트 (list)
# output : 레이블 데이터 (list)
def make_label_list(files_list):
    label_result = list()
    label_dict = {'hipnc': 0,
                'camera' : 1,
                'picture' : 2,
                'record' : 3,
                'stop' : 4,
                'end' : 5}

    for one in files_list:
        parsed_path = one.split('\\')
        if parsed_path[-2] in label_dict:
            label_result.append(label_dict[parsed_path[-2]])
        else:
            label_result.append(6)

    return label_result


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


# ## function : write_numpy_file
# about : 해당 경로의 wav 파일을 읽어서 int형 데이터 반환
# input : 파일 경로
# output : 해당 파일 데이터
def read_wav_file(**kwargs):
    if 'file_path' in kwargs.keys():
        file_path = kwargs['file_path']
    _, data = wavfile.read(file_path)
    return data








if __name__ == '__main__':
    print("hello, world~!!")
    files_list = list()
    files_list = find_files(filepath='D:\\voice_data_backup\\PNC_DB_ALL',
                file_ext='.wav')
    # print(files_list)

    train_data_set = list()
    label_list = list()

    label_list = make_label_list(files_list)

    for i, one_file in enumerate(files_list):

        one_data = read_wav_file(file_path=one_file)
        one_data = standardize_signal(one_data)

        result = cut_input_signal(one_data, frame_size=255, shift_size=128)

        # draw_graph(result)

        result = fit_determined_size(result)
        result = add_noise_data(result)

        train_data_set.append(result)

        print("\r{}th file is done...".format(i), end='')

    write_numpy_file(train_data_set,
                    label_numpy=np.array(label_list))


