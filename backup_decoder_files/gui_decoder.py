#%%
import sys
import random
import os
import threading

temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from draw_graph import result_graph as rg

from scipy import signal
from scipy.fft import fft, fftshift

import time
from scipy.io import wavfile

from sklearn.preprocessing import Normalizer

from analysis_signal import signal_trigger, util_module
from analysis_signal.util_module import standardization_func, new_minmax_normal, transpose_the_matrix
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
recording_time = 2
frame_t = 0.025
shift_t = 0.01
# buffer_s = 3000
buffer_s = 3000
voice_size = recording_time*sample_rate
chunk = 400
per_sec = sample_rate/chunk
num = 0
start_time, end_time = float(), float()


# label_dict = {0: 'None',
#             1: 'album',
#             2: 'back',
#             3: 'bright',
#             4: 'call',
#             5: 'camera',
#             6: 'dark',
#             7: 'end',
#             8: 'execute',
#             9: 'init',
#             10: 'picture',
#             11: 'position',
#             12: 'receive',
#             13: 'record',
#             14: 'reject',
#             15: 'stop',
#             16: 'hipnc'}


# label_dict = {0: 'None',
#             1: 'call', # 전화
#             2: 'camera', # 카메라
#             3: 'picture', # 촬영
#             4: 'record', # 녹화
#             5: 'stop', # 중지
#             6: 'hipnc'}
#
# label_dict = {0: 'None',
#             1: 'right answer'}

label_dict = {0: 'None',
            1: 'call', # 전화
            2: 'camera', # 카메라
            3: 'picture', # 촬영
            4: 'record', # 녹화
            5: 'stop', # 중지
            6: 'end'}  # 종료


## global model
num_label = 7
conv_shape = (199, 26, 1)

input_vec = tf.keras.Input(shape=conv_shape)
resnet = mr.residual_net_2D()
answer = resnet(input_vec, num_of_classes=num_label)
model = tf.keras.Model(inputs=input_vec, outputs=answer)

#%% epoch training loop
# h5_path_0 = 'D:\\resnet_model_only_train.h5'
# h5_path_best_0 = 'D:\\resnet_model_best_only_train.h5'
#
# h5_path_1 = 'D:\\resnet_model_all.h5'
# h5_path_best_1 = 'D:\\resnet_model_best_all.h5'

# parameter_h5 = 'D:\\new_ver_train_data\\keyword_model_parameter.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\keyword_model_parameter_3.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\keyword_model_parameter_4.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\keyword_model_parameter_6.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\keyword_model_parameter_7.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\keyword_model_parameter_8.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\keyword_model_parameter_9.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\keyword_model_parameter_10.h5'

# parameter_h5 = 'D:\\new_ver_train_data\\command_model_parameter.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\command_model_parameter_2.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\command_model_parameter_3.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\command_model_parameter_6.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\command_model_parameter_10.h5'
parameter_h5 = 'D:\\new_ver_train_data\\command_model_parameter_10_1.h5'

# parameter_h5 = 'D:\\new_ver_train_data\\call_model_parameter.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\call_model_parameter_8.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\call_model_parameter_10.h5'

# parameter_h5 = 'D:\\new_ver_train_data\\camera_model_parameter.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\camera_model_parameter_10.h5'

# parameter_h5 = 'D:\\new_ver_train_data\\picture_model_parameter.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\picture_model_parameter_10.h5'
# parameter_h5 = 'D:\\new_ver_train_data\\picture_model_parameter_8.h5'

# parameter_h5 = 'D:\\new_ver_train_data\\record_model_parameter.h5'

# parameter_h5 = 'D:\\new_ver_train_data\\stop_model_parameter.h5'

# parameter_h5 = 'D:\\new_ver_train_data\\end_model_parameter.h5'


# model.load_weights('D:\\resnet_model.h5')
# model.load_weights(h5_path_1)
model.load_weights(parameter_h5)


##
def receive_data(data, stack):
    # trigger_val = 0.5
    global start_time
    global num

    mean_val = np.mean(np.abs(data))

    stack.extend(data)
    if len(stack) > sample_rate*(recording_time+1):
        del stack[0:chunk]

    if float(trigger_val) <= float(mean_val) or num != 0:
        # print(num, mean_val)
        num+=1
        # print(num, mean_val)
        if num == 80:
            # print(num, mean_val)
            stack = util_module.standardization_func(stack)
            data = signal_trigger.evaluate_mean_of_frame(stack, frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*recording_time,
                                        threshold_value=0.5)

            data = data[:32000]
            start_time = time.time()
            decoding_wav_command(data)
            num = 0

    return


##
def send_data(chunk, stream):
    stack = list()
    while True:
        # for i in range(0, int(sr / chunk * seconds)):
        # data = stream.read(chunk)
        data = stream.read(chunk, exception_on_overflow = False)
        # print(data)
        # print(type(data))
        # print(len(data))
        data = np.frombuffer(data, 'int16')
        # print(data)
        # print(type(data))
        # print(len(data))

        # frames.append(data)
        # send = threading.Thread(target=None, name=None, args=(data,))
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

    # frames = []
    data = list()

    # data = [0, 0, 0, 0, 0]

    # time.sleep(0.2)
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
    logfb_feat = util_module.standardization_func(logfb_feat)

    train_feats = tf.expand_dims(logfb_feat, -1)
    print("data shape : "+ str(train_feats.shape))
    # conv_shape = (train_feats.shape[0], train_feats.shape[1], 1)
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


#%%
def decoding_wav_command(data):
    global end_time

    #%% loading data
    test_data, _  = generate_train_data(data)

    #%% build model
    # input_vec = tf.keras.Input(shape=conv_shape, batch_size=1)
    # model.summary()

    predictions = model.predict(test_data, verbose=1)

    # a = tf.math.argmax(predictions[0], axis=0)
    # a = np.argmax(predictions)
    # print(predictions[0])
    end_time = time.time()
    print(predictions)
    # # print(a)
    a = np.argmax(predictions)
    # print(predictions[0][a])
    # if predictions[0][a] > 0.8:
    #     print(label_dict[a])
    # elif a == 2:
    #     if predictions[0][a] > 0.50:
    #         print(label_dict[a])
    #     else:
    #         print(label_dict[0])
    # else:
    #     print(label_dict[0])
    # print(predictions.shape)

    print('\n')
    print(predictions[0][a])

    # if a == 0:
    #     temp = list()
    #     for i in range(1, num_label):
    #         if predictions[0][i] > float(1/(num_label-1)):
    #             temp.append([i, predictions[0][i]])
    #
    #     if len(temp) == 1:
    #         print(label_dict[temp[0][0]])
    #         print_result(temp[0][0], predictions)
    #     elif len(temp) > 1:
    #         temp_max = 0
    #         temp_label = 0
    #         for one in temp:
    #             if one[1] > temp_max:
    #                 temp_max = one[1]
    #                 temp_label = one[0]
    #         print(label_dict[temp_label])
    #         print_result(temp_label, predictions)
    #     else:
    #         print(label_dict[0])
    #         print_result(0, predictions)
    # else:
    #     # print(label_dict[a])
    #     if predictions[0][a] > 0.5:
    #         print(label_dict[a])
    #         print_result(a, predictions)
    #     elif a == 2:
    #         if predictions[0][a] > 0.50:
    #             print(label_dict[a])
    #             print_result(a, predictions)
    #         else:
    #             print(label_dict[0])
    #             print_result(0, predictions)
    #     else:
    #         print(label_dict[0])
    #         print_result(0, predictions)

    print("label : %d\tstring : %s" % (a, label_dict[a]))
    print("decoding time : %f" %(end_time-start_time))
    print('\n\n')

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
