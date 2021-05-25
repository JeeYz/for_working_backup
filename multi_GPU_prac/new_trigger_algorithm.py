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
tail_size = 8000
full_size = sr*4
trigger_val = 3.0

# ## function : find_files
# about : 파일을 탐색해 주는 함수
# input : 타겟 경로와 확장자
# output : 파일 리스트 
def find_files(filepath, file_ext):
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

# ## function : evaluate_mean_of_frame
# about : 입력되는 신호를 신호의 세기를 기준으로 잘라 주는 함수
# input : signal data, frame size, shift size
# output : 잘린 데이터
def evaluate_mean_of_frame(data, **kwargs):

    if "frame_time" in kwargs.keys():
        frame_size = kwargs['frame_size']
    if "shift_time" in kwargs.keys():
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

    result = data[temp:gap_frame_shift*end_index+tail_size]

    if full_size > len(result):
        result = fit_determined_size(result, full_size=full_size)
        result = add_noise_data(result, full_size=full_size)
    elif full_size == len(result):
        result = add_noise_data(result, full_size=full_size)
    else:
        result = result[:full_size]
        result = add_noise_data(result, full_size=full_size)

    return result


# ## function : fit_determined_size
# about : 잘려진 데이터의 길이를 정해진 길이에 맞춰 주는 함수
# input : 잘린 신호 데이터
# output : 정해진 길이로 맞춰진 데이터
def fit_determined_size(data, **kwargs):
    if "full_size" in kwargs.keys():
        full_size = kwargs['full_size']
    result = np.append(data, np.zeros(full_size-len(data)))

    return result


# ## function : add_noise_data
# about : 노이즈를 추가해 주는 데이터
# input : 신호 데이터
# output : 잡음이 추가된 데이터
def add_noise_data(data, **kwargs):
    if "full_size" in kwargs.keys():
        full_size = kwargs['full_size']

    noise_data = np.random.randn(full_size)*0.01
    result = data+noise_data

    return result


# ## function : draw_graph
# about : 신호 데이터를 그래프로 그려주는 함수
# input : 신호 데이터
# output : None
def draw_graph(data, **kwargs):

    

    return


# ## function : standardize_signal
# about : 입력된 신호 데이터를 표준화 시켜주는 함수
# input : 신호 데이터
# output : 표준화된 데이터
def standardize_signal(data, **kwargs):


    return


# ## function : write_file
# about : 정제된 데이터에 대해 numpy 혹은 wav 파일로 출력
# input : 데이터 (list)
# output : None
def write_file(data, **kwargs):
    return







if __name__ == '__main__':
    print("hello, world~!!")
