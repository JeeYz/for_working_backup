#%%
import threading

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import time

from tkinter import *
import pyaudio as pa

from python_speech_features import logfbank

pb_file = 'D:\\h5_pb_file\\saved_model.pb'
# tflite_file = 'D:\\h5_pb_file\\converted_model_resnet_all.tflite'
tflite_file = 'D:\\h5_pb_file\\converted_model_resnet_all_ver_3.tflite'

stack = list()
trigger_val = 150

sample_rate = 16000
recording_time = 2
frame_t = 0.025
shift_t = 0.01
buffer_s = 3000
voice_size = recording_time*sample_rate
chunk = 400
per_sec = sample_rate/chunk
num = 0
start_time, end_time = float(), float()

def standardization_func(data):
    return (data-np.mean(data))/np.std(data)

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

label_dict = {0: 'None',
            1: 'call', # 전화
            2: 'camera', # 카메라
            3: 'picture', # 촬영
            4: 'record', # 녹화
            5: 'stop', # 중지
            6: 'hipnc'}


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


##
def receive_data(data, stack):

    global start_time
    global num

    mean_val = np.mean(np.abs(data))

    stack.extend(data)
    if len(stack) > sample_rate*(recording_time+1):
        del stack[0:chunk]

    # print(len(stack))
    # print(num, mean_val)

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
                                        full_size = sample_rate*recording_time,
                                        threshold_value=0.5)
            data = data[:32000]
            # start_time = time.time()
            decoding_wav_command(data)
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

    max_number = 200

    data = np.asarray(data)

    logfb_feat = logfbank(data)
    logfb_feat = standardization_func(logfb_feat)

    logfb_feat = tf.keras.preprocessing.sequence.pad_sequences([logfb_feat],
                                            maxlen=max_number, padding='post', dtype='float32')

    train_feats = tf.expand_dims(logfb_feat, -1)
    # print("data shape : "+ str(train_feats.shape))
    conv_shape = (train_feats.shape[0], train_feats.shape[1], 1)
    conv_shape = (200, 26, 1)

    return np.asarray(train_feats), conv_shape


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
    num_label = 7

    #%% loading data

    temp_start = time.time()
    test_data, conv_shape = generate_train_data(data)
    test_data = np.array(test_data)
    print('================')
    print(time.time()-temp_start)
    print('================')

    # print(np.shape(test_data))
    # print(input_details)
    # print(interpreter.get_input_details())

    # temp_start = time.time()

    interpreter.set_tensor(input_details['index'], test_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details['index'])

    # print('================')
    # print(time.time()-temp_start)
    # print('================')

    end_time = time.time()
    # print(output_data)
    # print(a)
    a = np.argmax(output_data)
    print('\n')
    print(output_data[0][a])


    if a == 0:
        temp = list()
        for i in range(1, num_label):
            if output_data[0][i] > float(1/(num_label-1)):
                temp.append([i, output_data[0][i]])

        if len(temp) == 1:
            print(label_dict[temp[0][0]])
            print_result(temp[0][0], output_data)
        elif len(temp) > 1:
            temp_max = 0
            temp_label = 0
            for one in temp:
                if one[1] > temp_max:
                    temp_max = one[1]
                    temp_label = one[0]
            print(label_dict[temp_label])
            print_result(temp_label, output_data)
        else:
            print(label_dict[0])
            print_result(0, output_data)
    else:
        # print(label_dict[a])
        if output_data[0][a] > 0.5:
            print(label_dict[a])
            print_result(a, output_data)
        elif a == 2:
            if output_data[0][a] > 0.50:
                print(label_dict[a])
                print_result(a, output_data)
            else:
                print(label_dict[0])
                print_result(0, output_data)
        else:
            print(label_dict[0])
            print_result(0, output_data)


    # print(output_data.shape)
    print("decoding time : %f sec" %(end_time-start_time))
    print('\n')
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
