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

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

import matplotlib.pyplot as plt

import time
from scipy.io import wavfile

from sklearn.preprocessing import Normalizer

from tkinter import *
import pyaudio as pa
import wave


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


# flag = 0
keyword_label_dict = {0: 'None',
            1: 'hipnc'}

# flag = 1
command_label_dict = {0: 'None',
            1: 'camera', # 카메라
            2: 'picture', # 촬영
            3: 'record', # 녹화
            4: 'stop', # 중지
            5: 'end'} # 종료


# flag = 2
camera_label_dict = {0: 'None',
                1: 'camera'}

# flag = 3
picture_label_dict = {0: 'None',
                1: 'picture'}

# flag = 4
record_label_dict = {0: 'None',
                1: 'record'}

# flag = 5
stop_label_dict = {0: 'None',
                1: 'stop'}

# flag = 6
end_label_dict = {0: 'None',
                1: 'end'}


##
def standardization_func(data):
    return (data-np.mean(data))/np.std(data)


##
def normalization_func(data, max_val):
    return (data-np.min(data))/(max_val-np.min(data))


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

    # return result size of 32000
    return result[:32000]


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


# model
class simple_CNN_layers(layers.Layer):
    def __init__(self, **kwarg):
        super(simple_CNN_layers, self).__init__()

    def __call__(self, inputs, **kwarg):
        if 'num_of_classes' in kwarg.keys():
            number_of_classes = kwarg['num_of_classes']
        if 'input_shape' in kwarg.keys():
            input_shape = kwarg['input_shape']

        spectrogram = tf.signal.stft(inputs, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, -1)
        # spectrogram = np.asarray(spectrogram)
        spectrogram = tf.stack(spectrogram)
        print("\n******************************\n")
        print(spectrogram.shape, spectrogram)
        print("\n******************************\n")
        
        # input_vec = tf.keras.Input(shape=(spectrogram.shape[1], spectrogram.shape[2], 1))
        # x = preprocessing.Resizing(64, 64)(input_vec)
        x = preprocessing.Resizing(32, 32)(spectrogram)
        x = preprocessing.Normalization()(x)
        # x = preprocessing.Normalization()(x)
        # x = layers.Conv2D(32, 3, activation='relu')(input_vec)
        x = layers.Conv2D(32, 3, activation='relu')(x)
        # x = layers.Conv2D(32, 3, activation='relu')(x)
        # x = layers.MaxPooling2D()(x)
        # x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        answer = layers.Dense(number_of_classes, activation='softmax')(x)

        return answer



## global model
conv_shape = (249, 129, 1)
global_flag = 1

## keyword global model
if global_flag == 0:
    keyword_label_num = 2
    keyword_input = tf.keras.Input(shape=conv_shape)
    simple_cnn = simple_CNN_layers()
    keyword_answer = simple_cnn(keyword_input, num_of_classes=keyword_label_num)
    keyword_model = tf.keras.Model(inputs=keyword_input, outputs=keyword_answer)
    keyword_h5 = 'D:\\new_ver_train_data\\keyword_model_parameter.h5'
    keyword_h5_best = 'D:\\new_ver_train_data\\keyword_model_parameter_best.h5'
    keyword_model.load_weights(keyword_h5)
    

## global command
elif global_flag == 1:
    command_label_num = 6
    simple_cnn = simple_CNN_layers()
    command_input = tf.keras.Input(shape=(32000,))
    command_answer = simple_cnn(command_input, num_of_classes=command_label_num)
    command_model = tf.keras.Model(inputs=command_input, outputs=command_answer)
    # command_h5 = 'D:\\new_ver_train_data\\command_model_parameter.h5'
    command_h5 = 'D:\\example_STFT.h5'
    command_model.load_weights(command_h5)


## camera confirm
elif global_flag == 2:
    camera_label_num = 2
    camera_input = tf.keras.Input(shape=conv_shape)
    camera_resnet = mr.residual_net_2D()
    camera_answer = camera_resnet(camera_input, num_of_classes=camera_label_num)
    camera_model = tf.keras.Model(inputs=camera_input, outputs=camera_answer)
    call_h5 = 'D:\\new_ver_train_data\\call_model_parameter.h5'
    call_h5_best = 'D:\\new_ver_train_data\\call_model_parameter_best.h5'
    call_model.load_weights(call_h5)

## picture confirm
elif global_flag == 3:
    picture_label_num = 2
    picture_input = tf.keras.Input(shape=conv_shape)
    picture_resnet = mr.residual_net_2D()
    picture_answer = picture_resnet(picture_input, num_of_classes=picture_label_num)
    picture_model = tf.keras.Model(inputs=picture_input, outputs=picture_answer)
    picture_h5 = 'D:\\new_ver_train_data\\picture_model_parameter.h5'
    picture_h5_best = 'D:\\new_ver_train_data\\picture_model_parameter_best.h5'
    picture_model.load_weights(picture_h5)

## record confirm
elif global_flag == 4:
    record_label_num = 2
    record_input = tf.keras.Input(shape=conv_shape)
    record_resnet = mr.residual_net_2D()
    record_answer = record_resnet(record_input, num_of_classes=record_label_num)
    record_model = tf.keras.Model(inputs=record_input, outputs=record_answer)
    record_h5 = 'D:\\new_ver_train_data\\record_model_parameter.h5'
    record_h5_best = 'D:\\new_ver_train_data\\record_model_parameter_best.h5'
    record_model.load_weights(record_h5)

# ## stop confirm
elif global_flag == 5:
    stop_label_num = 2
    stop_input = tf.keras.Input(shape=conv_shape)
    stop_resnet = mr.residual_net_2D()
    stop_answer = stop_resnet(stop_input, num_of_classes=stop_label_num)
    stop_model = tf.keras.Model(inputs=stop_input, outputs=stop_answer)
    stop_h5 = 'D:\\new_ver_train_data\\stop_model_parameter.h5'
    stop_h5_best = 'D:\\new_ver_train_data\\stop_model_parameter_best.h5'
    stop_model.load_weights(stop_h5)
 
# ## end confirm
elif global_flag == 6:
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

    if float(trigger_val) <= float(mean_val) or num != 0:
        num+=1
        if num == 120:
            start_time = time.time()
            stack = standardization_func(stack)
            data = evaluate_mean_of_frame(stack, frame_time=frame_t,
                                            shift_time=shift_t,
                                            sample_rate=sample_rate,
                                            buffer_size=buffer_s,
                                            full_size = sample_rate*2,
                                            threshold_value=0.5)
            
            print("+++++shape : {shape}".format(shape=data.shape))

            decoding_voice(data)

            num = 0

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
def print_result(index_num, output_data):

    for i, j in enumerate(output_data[0]):
        if i == index_num:
            print("%d : %6.2f %% <<< %s"%(i, j*100, command_label_dict[index_num]))
        else:
            print("%d : %6.2f %%"%(i, j*100))

    return


##
def decoding_voice(test_data):
    global end_time, global_flag, command_model

    print("hello, world~!!")

    # time.sleep(1000)
    predictions = command_model.predict_step(test_data)
    end_time = time.time()

    a = np.argmax(predictions)

    print('\n')
    print(predictions[0][a])

    print("label : {label}, prediction score : {pre}".format(label=a, pre=predictions[0][a]))

    print_result(a, predictions)
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



#%%
def main():

    record_voice()








#%%
if __name__=='__main__':
    main()





## endl
