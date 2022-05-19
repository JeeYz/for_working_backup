#%%
'''
# decoder for command speech
'''

#%%
import sys
sys.path.append("D:\\Programming\\code\\girl_from_ipanema")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import basic_cnn_block as bcblock
import residual_block as resnet_block

from tkinter import *
import pyaudio as pa
import numpy as np
import wave
import time
from scipy.io import wavfile
import matplotlib.pyplot as plt

from python_speech_features import mfcc
from python_speech_features import logfbank
from preprocessing_for_data import new_minmax_normal

h5_path = 'D:\\Programming\\for_decoding\\basic_cnn_model_best.h5'
h5_path1 = 'D:\\Programming\\for_decoding\\basic_cnn_command_model_best.h5'
h5_path2 = 'D:\\Programming\\for_decoding\\basic_cnn_command_model.h5'

# command parameter ResNet
h5_path3 = 'D:\\Programming\\for_decoding\\resnet_model_best.h5'
h5_path4 = 'D:\\Programming\\for_decoding\\resnet_model.h5'


wav_path = 'C:\\Users\\jyback_pnc\\Desktop\\for_decoding\\keyword.wav'
wav_path1 = 'C:\\Users\\jyback_pnc\\Desktop\\for_decoding\\command.wav'

# test file from training
wav_path2 = 'C:\\Users\\jyback_pnc\\Desktop\\for_decoding\\command_1.wav'
wav_path3 = 'C:\\Users\\jyback_pnc\\Desktop\\for_decoding\\command_2.wav'
wav_path4 = 'C:\\Users\\jyback_pnc\\Desktop\\for_decoding\\command_3.wav'
wav_path5 = 'C:\\Users\\jyback_pnc\\Desktop\\for_decoding\\command_4.wav'
wav_path6 = 'D:\Programming\for_decoding\\command_5.wav'
wav_path7 = 'C:\\Users\\jyback_pnc\\Desktop\\for_decoding\\command_6.wav'
wav_path8 = 'D:\\Programming\\for_decoding\\command_7.wav'


def detect_wav_file(data):
    rate_max = 0.37
    add_size = 3000

    temp_max = np.max(data)
    print(temp_max)
    for i,j in enumerate(data):
        if temp_max*rate_max < j:
            start_index = i
            break

    for i,j in enumerate(reversed(data)):
        if temp_max*rate_max < j:
            end_index = len(data)-i
            break

    print(start_index, end_index)
    result = data[start_index:end_index]

    add_data = np.random.rand(add_size)

    temp = np.append(add_data, result)
    result = np.append(temp, add_data)

    return result

#%%
def record_voice_command():
    chunk = 2**10
    sample_format = pa.paInt16
    channels = 1
    sr = 16000

    seconds = 2

    filename = wav_path1

    p = pa.PyAudio()

    # recording
    print('Recording')

    stream = p.open(format=sample_format, channels=channels, rate=sr,
                    frames_per_buffer=chunk, input=True)

    frames = []

    # time.sleep(0.2)

    for i in range(0, int(sr / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(sr)
    wf.writeframes(b''.join(frames))
    wf.close()

    p.terminate()

    decoding_wav_command()

    return


#%%
def print_status():

    return

#%%
def read_wav_command():

    # start_point = 10000

    fs, data = wavfile.read(wav_path8)

    # plt.plot(data)
    # plt.show()
    #
    data = detect_wav_file(data)
    #
    # plt.plot(data)
    # plt.show()
    # data = data[start_point:start_point+15000]

    data = logfbank(data, fs)
    data = new_minmax_normal(data)

    # data = tf.expand_dims(data, 0)
    data = tf.expand_dims(data, -1)

    print("***** : ", data.shape)

    return data



#%%
def decoding_wav_command():

    num_label = 16

    input_data = read_wav_command()
    # input_data = tf.expand_dims(input_data, -1)
    # conv_shape = input_data.shape
    conv_shape = (input_data.shape[0], input_data.shape[1], 1)
    print(conv_shape)
    print(input_data.shape)
    input_vec = tf.keras.Input(shape=conv_shape, batch_size=1)

    resnet = resnet_block.residual_net_2D(pooling_bool=False, init_channels=128)

    answer = resnet(input_vec, num_of_classes=num_label, dense_softmax=True)
    print('^^^^^^^^^', answer.shape)
    model = tf.keras.Model(inputs=input_vec, outputs=answer)

    model.summary()

    model.load_weights(h5_path3)

    input_data = tf.expand_dims(input_data, 0)

    predictions = model.predict(input_data, verbose=1)

    # a = tf.math.argmax(predictions[0], axis=0)
    # a = np.argmax(predictions)
    # print(predictions[0])

    print(predictions)
    # print(a)
    a = np.argmax(predictions)

    print(a)
    print(predictions.shape)


#%%
def main():
    root = Tk()
    root.geometry('250x250')
    root.title('decoder')

    my_label = Label(root, text = "Recording and Print result")
    my_label.pack(pady = 10)

    my_button1 = Button(root, text = "Command_Record", command = record_voice_command, width = 20)
    my_button1.pack(pady = 10)

    my_button2 = Button(root, text = "Decoding", command = decoding_wav_command, width = 10)
    my_button2.pack(pady = 10)


    root.mainloop()


#%%
if __name__=='__main__':
    main()
