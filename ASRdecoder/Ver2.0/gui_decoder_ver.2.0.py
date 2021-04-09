#%%
import sys
import random
import os
import threading

temp = __file__.split('\\')
temp = '\\'.join(temp[:-3])
sys.path.append(temp)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import time
from scipy.io import wavfile

from sklearn.preprocessing import Normalizer
from ASRdecoder import model_resnet as mr

from tkinter import *
import pyaudio as pa
import wave

from python_speech_features import logfbank
from preprocessing_for_data import new_minmax_normal

train_files_name = 'D:\\train_data_files.txt'
test_files_name = 'D:\\test_data_files.txt'

mod_train_files_name = 'D:\\mod_train_data_files.txt'
mod_test_files_name = 'D:\\mod_test_data_files.txt'

mod_full_data_files_name = 'D:\\mod_full_data_files_list.txt'
mod_full_data_files_name_shuffle = 'D:\\mod_full_data_files_list_shuffle.txt'

mod_train_data_path = 'D:\\mod_train_data.npz'
mod_test_data_path = 'D:\\mod_test_data.npz'

stack = list()
trigger_val = 150

sample_rate = 16000
recording_time = 3
frame_t = 0.025
shift_t = 0.01
buffer_s = 3000
voice_size = 2*sample_rate
chunk = 400
per_sec = sample_rate/chunk
num = 0
start_time, end_time = float(), float()
global_flag = 0


keyword_label_dict = {0: 'None',
            1: 'hipnc'}

command_label_dict = {0: 'None',
            1: 'call', # 전화
            2: 'camera', # 카메라
            3: 'picture', # 촬영
            4: 'record', # 녹화
            5: 'stop', # 중지
            6: 'end'} # 종료

call_label_dict = {0: 'None',
                1: 'call'}

camera_label_dict = {0: 'None',
                1: 'camera'}

picture_label_dict = {0: 'None',
                1: 'picture'}

record_label_dict = {0: 'None',
                1: 'record'}

stop_label_dict = {0: 'None',
                1: 'stop'}

end_label_dict = {0: 'None',
                1: 'end'}


##
def standardization_func(data):
    return (data-np.mean(data))/np.std(data)


##
def evaluate_mean_of_frame(data, **kwargs):

    if "frame_time" in kwargs.keys():
        frame_time = kwargs['frame_time']
    if "shift_time" in kwargs.keys():
        shift_time = kwargs['shift_time']
    if "sample_rate" in kwargs.keys():
        sr = kwargs['sample_rate']
    if "buffer_size" in kwargs.keys():
        buf_size = kwargs['buffer_size']
    if "full_size" in kwargs.keys():
        full_size = kwargs['full_size']
    if "threshold_value" in kwargs.keys():
        trigger_val = kwargs['threshold_value']

    frame_size = int(sr*frame_time)
    shift_size = int(sr*shift_time)

    num_frames = len(data)//(frame_size-shift_size)+1

    mean_val_list = list()

    for i in range(num_frames):
        temp_n = i*(frame_size-shift_size)
        if temp_n+frame_size > len(data):
            one_frame_data = data[temp_n:len(data)]
        else:
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
            end_index = len(mean_val_list)-i
            break
        else:
            end_index = len(mean_val_list)

    temp = (frame_size-shift_size)*start_index-buf_size
    if temp <= 0:
        temp = 0

    result = data[temp:(frame_size-shift_size)*end_index+buf_size]

    if full_size > len(result):
        result = fit_determined_size(result, full_size=full_size)
        result = add_noise_data(result, full_size=full_size)
    elif full_size == len(result):
        result = add_noise_data(result, full_size=full_size)
    else:
        result = result[:full_size]
        result = add_noise_data(result, full_size=full_size)

    return result


def fit_determined_size(data, **kwargs):
    if "full_size" in kwargs.keys():
        full_size = kwargs['full_size']
    result = np.append(data, np.zeros(full_size-len(data)))

    return result


##
def add_noise_data(data, **kwargs):
    if "full_size" in kwargs.keys():
        full_size = kwargs['full_size']

    noise_data = np.random.randn(full_size)*0.01
    result = data+noise_data

    return result


## global model
conv_shape = (199, 26, 1)

## keyword global model
keyword_label_num = 2
keyword_input = tf.keras.Input(shape=conv_shape)
keyword_resnet = mr.residual_net_2D()
keyword_answer = keyword_resnet(keyword_input, num_of_classes=keyword_label_num)
keyword_model = tf.keras.Model(inputs=keyword_input, outputs=keyword_answer)
keyword_h5 = 'D:\\new_ver_train_data\\keyword_model_parameter.h5'
keyword_h5_best = 'D:\\new_ver_train_data\\keyword_model_parameter_best.h5'
keyword_model.load_weights(keyword_h5)

## global command
command_label_num = 7
command_input = tf.keras.Input(shape=conv_shape)
command_resnet = mr.residual_net_2D()
command_answer = command_resnet(command_input, num_of_classes=command_label_num)
command_model = tf.keras.Model(inputs=command_input, outputs=command_answer)
command_h5 = 'D:\\new_ver_train_data\\command_model_parameter.h5'
command_h5_best = 'D:\\new_ver_train_data\\command_model_parameter_best.h5'
command_model.load_weights(command_h5)

## call confirm
call_label_num = 2
call_input = tf.keras.Input(shape=conv_shape)
call_resnet = mr.residual_net_2D()
call_answer = call_resnet(call_input, num_of_classes=call_label_num)
call_model = tf.keras.Model(inputs=call_input, outputs=call_answer)
call_h5 = 'D:\\new_ver_train_data\\call_model_parameter.h5'
call_h5_best = 'D:\\new_ver_train_data\\call_model_parameter_best.h5'
call_model.load_weights(call_h5)

## camera confirm
camera_label_num = 2
camera_input = tf.keras.Input(shape=conv_shape)
camera_resnet = mr.residual_net_2D()
camera_answer = camera_resnet(camera_input, num_of_classes=camera_label_num)
camera_model = tf.keras.Model(inputs=camera_input, outputs=camera_answer)
call_h5 = 'D:\\new_ver_train_data\\call_model_parameter.h5'
call_h5_best = 'D:\\new_ver_train_data\\call_model_parameter_best.h5'
call_model.load_weights(call_h5)

## picture confirm
picture_label_num = 2
picture_input = tf.keras.Input(shape=conv_shape)
picture_resnet = mr.residual_net_2D()
picture_answer = picture_resnet(picture_input, num_of_classes=picture_label_num)
picture_model = tf.keras.Model(inputs=picture_input, outputs=picture_answer)
picture_h5 = 'D:\\new_ver_train_data\\picture_model_parameter.h5'
picture_h5_best = 'D:\\new_ver_train_data\\picture_model_parameter_best.h5'
picture_model.load_weights(picture_h5)

## record confirm
record_label_num = 2
record_input = tf.keras.Input(shape=conv_shape)
record_resnet = mr.residual_net_2D()
record_answer = record_resnet(record_input, num_of_classes=record_label_num)
record_model = tf.keras.Model(inputs=record_input, outputs=record_answer)
record_h5 = 'D:\\new_ver_train_data\\record_model_parameter.h5'
record_h5_best = 'D:\\new_ver_train_data\\record_model_parameter_best.h5'
record_model.load_weights(record_h5)

## stop confirm
stop_label_num = 2
stop_input = tf.keras.Input(shape=conv_shape)
stop_resnet = mr.residual_net_2D()
stop_answer = stop_resnet(stop_input, num_of_classes=stop_label_num)
stop_model = tf.keras.Model(inputs=stop_input, outputs=stop_answer)
stop_h5 = 'D:\\new_ver_train_data\\stop_model_parameter.h5'
stop_h5_best = 'D:\\new_ver_train_data\\stop_model_parameter_best.h5'
stop_model.load_weights(stop_h5)

## end confirm
end_label_num = 2
end_input = tf.keras.Input(shape=conv_shape)
end_resnet = mr.residual_net_2D()
end_answer = end_resnet(end_input, num_of_classes=end_label_num)
end_model = tf.keras.Model(inputs=end_input, outputs=end_answer)
end_h5 = 'D:\\new_ver_train_data\\end_model_parameter.h5'
end_h5_best = 'D:\\new_ver_train_data\\end_model_parameter_best.h5'
end_model.load_weights(end_h5)


##
def receive_data(data, stack):

    global start_time
    global num, global_flag

    mean_val = np.mean(np.abs(data))

    stack.extend(data)

    if len(stack) > sample_rate*(recording_time+1):
        del stack[0:chunk]

    if global_flag == 0:
        if float(trigger_val) <= float(mean_val) or num != 0:
            # print(num, mean_val)
            num+=1
            # print(num, mean_val)
            if num == 80:
                # print(num, mean_val)
                start_time = time.time()
                stack = standardization_func(stack)
                data = evaluate_mean_of_frame(stack, frame_time=frame_t,
                                            shift_time=shift_t,
                                            sample_rate=sample_rate,
                                            buffer_size=buffer_s,
                                            full_size = sample_rate*2,
                                            threshold_value=0.5)


                test_data, _  = generate_train_data(data)
                decoding_keyword(test_data)
                num = 0

    elif global_flag ==1:
        num+=1
        if num == 120:
            # print(num, mean_val)
            start_time = time.time()
            stack = standardization_func(stack)
            data = evaluate_mean_of_frame(stack, frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*2,
                                        threshold_value=0.5)


            test_data, _  = generate_train_data(data)
            decoding_command(test_data)
            num = 0

    else:
        print("error!!")
        time.sleep(1000)


    return


##
def send_data(chunk, stream):
    stack = list()
    while True:
        data = stream.read(chunk, exception_on_overflow = False)
        data = np.frombuffer(data, 'int16')
        receive_data(data, stack)

    return

##
def record_voice():

    chunk = 400
    sample_format = pa.paInt16
    channels = 1
    sr = 16000

    seconds = 1

    p = pa.PyAudio()

    # recording
    print('Recording')

    stream = p.open(format=sample_format, channels=channels, rate=sr,
                    frames_per_buffer=chunk, input=True)

    data = list()

    send = threading.Thread(target=send_data, args=(chunk, stream))
    decoder = threading.Thread(target=receive_data, args=(data, stack))

    send.start()
    decoder.start()

    while True:
        time.sleep(0.1)

    send.join()
    decoder.join()
    draw.join()

    stream.stop_stream()
    stream.close()

    return


##
def generate_train_data(data):

    data = np.asarray(data)

    logfb_feat = logfbank(data)
    logfb_feat = standardization_func(logfb_feat)

    train_feats = tf.expand_dims(logfb_feat, -1)
    print("data shape : "+ str(train_feats.shape))

    conv_shape = (199, 26, 1)

    return np.asarray([train_feats]), conv_shape


##
def print_result(index_num, output_data):

    for i, j in enumerate(output_data[0]):
        if i == index_num:
            print("%d : %6.2f %% <<< %s"%(i, j*100, label_dict[index_num]))
        else:
            print("%d : %6.2f %%"%(i, j*100))

    return


##
def decoding_keyword(test_data):
    global end_time, global_flag
    print("hello, world~!!")
    predictions = keyword_model.predict(test_data, verbose=1)
    end_time = time.time()

    a = np.argmax(predictions)

    print('\n')
    print(predictions[0][a])

    if a == 0:
        print("label : %d\t\tglobal flag : %d"%(a, global_flag))
        print("it's not keyword...")
    else:
        global_flag = 1
        print("label : %d\t\tglobal flag : %d"%(a, global_flag))
        print("please, input command...")

    print("decoding time : %f" %(end_time-start_time))
    return


##
def return_label_number(predictions):
    label_number = 0
    max_temp = 0
    for i,j in enumerate(predictions):
        if i == 0:
            continue
        else:
            if j > max_temp:
                max_temp = j
                label_number = i

    return label_number


##
def decoding_command(test_data):
    global end_time, global_flag

    predictions = command_model.predict(test_data, verbose=1)

    a = np.argmax(predictions)

    print('\n')
    print(predictions[0][a])
    print(predictions[0])

    if a == 0:
        print("label : %d\t\tglobal flag : %d"%(a, global_flag))
        print(command_label_dict[a])
        if predictions[0][a] != 1.0:
            label_para = return_label_number(predictions[0])
            confirm_command(test_data, label_para)
        global_flag = 0
    else:
        print("label : %d\t\tglobal flag : %d"%(a, global_flag))
        print(command_label_dict[a])
        global_flag = 0


    end_time = time.time()

    print("decoding time : %f" %(end_time-start_time))

    return


##
def confirm_command(test_data, label_para):
    global end_time, global_flag

    if label_para == 1:
        predictions = call_model.predict(test_data, verbose=1)
    elif label_para == 2:
        predictions = camera_model.predict(test_data, verbose=1)
    elif label_para == 3:
        predictions = picture_model.predict(test_data, verbose=1)
    elif label_para == 4:
        predictions = record_model.predict(test_data, verbose=1)
    elif label_para == 5:
        predictions = stop_model.predict(test_data, verbose=1)
    elif label_para == 6:
        predictions = end_model.predict(test_data, verbose=1)

    a = np.argmax(predictions)

    print('\n')
    print(predictions[0][a])


    if a == 0:
        print("label : %d\t\tglobal flag : %d"%(a, global_flag))
        print(command_label_dict[a])
    else:
        print("label : %d\t\tglobal flag : %d"%(a, global_flag))
        print(command_label_dict[label_para])

    global_flag = 0

    return


#%%
def decoding_wav_command(data):
#     global end_time
#
#     #%% loading data
#     test_data, _  = generate_train_data(data)
#
#     predictions = model.predict(test_data, verbose=1)
#     end_time = time.time()
#
#     a = np.argmax(predictions)
#
#     print('\n')
#     print(predictions[0][a])
#
#
#     if a == 0:
#         temp = list()
#         for i in range(1, num_label):
#             if predictions[0][i] > float(1/(num_label-1)):
#                 temp.append([i, predictions[0][i]])
#
#         if len(temp) == 1:
#             print(label_dict[temp[0][0]])
#             print_result(temp[0][0], predictions)
#         elif len(temp) > 1:
#             temp_max = 0
#             temp_label = 0
#             for one in temp:
#                 if one[1] > temp_max:
#                     temp_max = one[1]
#                     temp_label = one[0]
#             print(label_dict[temp_label])
#             print_result(temp_label, predictions)
#         else:
#             print(label_dict[0])
#             print_result(0, predictions)
#     else:
#         # print(label_dict[a])
#         if predictions[0][a] > 0.5:
#             print(label_dict[a])
#             print_result(a, predictions)
#         elif a == 2:
#             if predictions[0][a] > 0.50:
#                 print(label_dict[a])
#                 print_result(a, predictions)
#             else:
#                 print(label_dict[0])
#                 print_result(0, predictions)
#         else:
#             print(label_dict[0])
#             print_result(0, predictions)
#
#     print("decoding time : %f" %(end_time-start_time))
#
    return


#%%
def main():
    # root = Tk()
    # root.geometry('250x250')
    # root.title('decoder')
    #
    # my_label = Label(root, text = "Recording and Print result")
    # my_label.pack(pady = 10)
    #
    # my_button2 = Button(root, text = "Decoding", command = record_voice, width = 10)
    # my_button2.pack(pady = 10)
    #
    #
    # root.mainloop()

    record_voice()








#%%
if __name__=='__main__':
    main()





## endl
