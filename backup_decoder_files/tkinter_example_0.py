# -*- coding: utf-8 -*-

#%%
import sys
sys.path.append("C:\\Users\\jyback_pnc\\Desktop\\for_decoding")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import basic_cnn_block as bcblock

from tkinter import *
import pyaudio as pa
import numpy as np
import wave
import time
from scipy.io import wavfile

from python_speech_features import mfcc
from python_speech_features import logfbank
from preprocessing_for_data import new_minmax_normal

h5_path = 'C:\\Users\\jyback_pnc\\Desktop\\for_decoding\\basic_cnn_model_best.h5'
h5_path1 = 'C:\\Users\\jyback_pnc\\Desktop\\for_decoding\\basic_cnn_command_model_best.h5'
h5_path2 = 'C:\\Users\\jyback_pnc\\Desktop\\for_decoding\\basic_cnn_command_model.h5'
wav_path = 'C:\\Users\\jyback_pnc\\Desktop\\for_decoding\\keyword.wav'
wav_path1 = 'C:\\Users\\jyback_pnc\\Desktop\\for_decoding\\command.wav'

#%%
def record_voice_1():
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

    decoding_wav_1()
    return



#%%
def record_voice():
    chunk = 2**10
    sample_format = pa.paInt16
    channels = 1
    sr = 16000

    seconds = 2

    filename = wav_path

    p = pa.PyAudio()

    # recording
    print('Recording')

    stream = p.open(format=sample_format, channels=channels, rate=sr,
                    frames_per_buffer=chunk, input=True)

    frames = []

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

    decoding_wav()
    return


#%%
def print_status():

    return


#%%
def read_wav_1():

    fs, data = wavfile.read(wav_path1)
    data = logfbank(data, fs)
    data = new_minmax_normal(data)
    # data = [data]
    # data = np.expand_dims(data, axis=0)
    # data = np.expand_dims(data, axis=-1)
    # data = np.expand_dims(data, axis=0)
    data = tf.expand_dims(data, 0)
    data = tf.expand_dims(data, -1)
    print(data.shape)

    return data


#%%
def read_wav():

    fs, data = wavfile.read(wav_path)
    data = logfbank(data, fs)
    data = new_minmax_normal(data)
    # data = [data]
    # data = np.expand_dims(data, axis=0)
    # data = np.expand_dims(data, axis=-1)
    # data = np.expand_dims(data, axis=0)
    data = tf.expand_dims(data, 0)
    data = tf.expand_dims(data, -1)
    print(data.shape)

    return data

#%%
def decoding_wav_1():

    num_label = 16

    input_data = read_wav_1()
    # input_data = tf.expand_dims(input_data, -1)
    # conv_shape = input_data.shape
    conv_shape = (input_data.shape[0], input_data.shape[1], 1)
    input_vec = tf.keras.Input(shape=conv_shape)

    basic_cnn = bcblock.norm_cnn(channels_size=32, dropout_value=0.1,
                                 num_of_blocks=3, activation='relu',
                                 # input_shape=conv_shape,
                                 pooling_bool=True, kernel_size=(3, 3))

    answer = basic_cnn(input_vec, num_of_classes=num_label, dense_softmax=True)

    model = tf.keras.Model(inputs=input_vec, outputs=answer)

    # model.summary()

    model.load_weights(h5_path1)

    predictions = model.predict(input_data, batch_size=1)
    # a = tf.math.argmax(predictions[0], axis=0)
    a = np.argmax(predictions[0], axis=0)
    print(predictions)
    print(a)
    print(predictions.shape)



#%%
def decoding_wav():

    num_label = 2

    input_data = read_wav()
    # input_data = tf.expand_dims(input_data, -1)
    # conv_shape = input_data.shape
    conv_shape = (input_data.shape[0], input_data.shape[1], 1)
    input_vec = tf.keras.Input(shape=conv_shape)

    basic_cnn = bcblock.norm_cnn(channels_size=32, dropout_value=0.1,
                                 num_of_blocks=3, activation='relu',
                                 # input_shape=conv_shape,
                                 pooling_bool=True, kernel_size=(3, 3))

    answer = basic_cnn(input_vec, num_of_classes=num_label, dense_softmax=True)

    model = tf.keras.Model(inputs=input_vec, outputs=answer)

    # model.summary()

    model.load_weights(h5_path)

    predictions = model.predict(input_data, batch_size=1)
    # a = tf.math.argmax(predictions[0], axis=0)
    a = np.argmax(predictions[0], axis=0)
    print(predictions)
    if predictions[0][1] > 0.70:
        print(a)
    else:
        print(0)
    # print(predictions.shape)


#%%
def main():
    root = Tk()
    root.geometry('250x250')
    root.title('decoder')

    my_label = Label(root, text = "Recording and Print result")
    my_label.pack(pady = 10)

    my_button0 = Button(root, text = "Keyword_Record", command = record_voice, width = 20)
    my_button0.pack(pady = 10)

    my_button1 = Button(root, text = "Command_Record", command = record_voice_1, width = 20)
    my_button1.pack(pady = 10)

    my_button2 = Button(root, text = "Decoding", command = decoding_wav_1, width = 10)
    my_button2.pack(pady = 10)


    root.mainloop()


#%%
if __name__=='__main__':
    main()
