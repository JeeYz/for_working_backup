#%% import sklearn
import os
import sys
temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing

#%% standardization

import pyaudio as pa
from scipy.io import wavfile
import wave

import numpy as np
import time
import matplotlib.pyplot as plt

from python_speech_features import mfcc
from python_speech_features import logfbank
# from analysis_signal.util_module import standardization_func, new_minmax_normal, transpose_the_matrix


def standardization_func(data):
    return (data-np.mean(data))/np.std(data)

#%%
def draw_graph_raw_signal(data, **kwargs):

    title_name = kwargs['title_name']

    plt.figure()
    plt.plot(data)

    plt.xlabel('sample rate')
    plt.ylabel('amplitude')
    plt.title(title_name)

    # plt.tight_layout()
    # plt.show()

    return



##%
def draw_graph_logfbank(data, sr, **kwargs):
    title_name = kwargs['title_name']

    # data = logfbank(data, sr)
    data = transpose_the_matrix(data)

    plt.figure()
    plt.plot()
    plt.pcolormesh(data)
    # plt.plot(data_0)

    plt.xlabel('frame sequence')
    plt.ylabel('number of filters')
    plt.title(title_name)

    plt.tight_layout()
    plt.colorbar()
    # plt.show()

    return



#%%
def draw_graph_logfbank_norm(data, sr, **kwargs):

    if "title_name" in kwargs.keys():
        title_name = kwargs['title_name']

    data = logfbank(data, sr)
    data = transpose_the_matrix(data)

    data = new_minmax_normal([data])

    plt.figure()
    plt.plot()
    plt.pcolormesh(data[0])
    # plt.plot(data_0)

    plt.xlabel('frame sequence')
    plt.ylabel('number of filters')
    plt.title(title_name)
    plt.colorbar()
    plt.tight_layout()

    return








## main
if __name__ == '__main__':

    # chunk = 400
    # sample_format = pa.paInt16
    # channels = 1
    # sr = 16000
    # FORMAT = pa.paInt16
    #
    # seconds = 10
    #
    # p = pa.PyAudio()
    #
    # # recording
    # print('Recording')
    #
    # stream = p.open(format=sample_format, channels=channels, rate=sr,
    #                 frames_per_buffer=chunk, input=True)
    #
    # # frames = []
    # data = list()
    #
    # for i in range(int(sr/chunk*seconds)):
    #     temp = stream.read(chunk, exception_on_overflow = False)
    #     temp = np.frombuffer(temp, 'int16')
    #     data.append(temp)
    #     # print(len(temp))
    #     # data.extend(temp)
    #
    # waveFile = wave.open('output_0.wav', 'wb')
    #
    # waveFile.setnchannels(channels)
    #
    # waveFile.setsampwidth(p.get_sample_size(FORMAT))
    #
    # waveFile.setframerate(sr)
    #
    # waveFile.writeframes(b''.join(data))
    #
    # waveFile.close()
    #
    # sampler, data = wavfile.read('output_0.wav')
    # print(data)
    # data = standardization_func(data)
    # draw_graph_raw_signal(data, title_name='raw')
    #
    # stream = p.open(format=sample_format, channels=channels, rate=sr,
    #                 frames_per_buffer=chunk, input=True)
    #
    # # frames = []
    #
    # print('Recording')
    # data = list()
    #
    # for i in range(int(sr/chunk*seconds)):
    #     temp = stream.read(chunk, exception_on_overflow = False)
    #     temp = np.frombuffer(temp, 'int16')
    #     data.append(temp)
    #     # print(len(temp))
    #     # data.extend(temp)
    #
    # waveFile = wave.open('output_1.wav', 'wb')
    #
    # waveFile.setnchannels(channels)
    #
    # waveFile.setsampwidth(p.get_sample_size(FORMAT))
    #
    # waveFile.setframerate(sr)
    #
    # waveFile.writeframes(b''.join(data))
    #
    # waveFile.close()

    all_norm_data = list()

    for (path, dir, files) in os.walk("../models/GEN_traindata_Ver.1.0_CR_Ver.2.4/"):   
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == ".wav":
                file_name = path + "\\" + filename
                all_norm_data.append(file_name)

    for one in all_norm_data:

        sampler, data = wavfile.read(one)
        # data = standardization_func(data)
        draw_graph_raw_signal(data, title_name='raw')

        plt.show()

    # stream.stop_stream()
    # stream.close()
    #
    # p.terminate()




## endl
