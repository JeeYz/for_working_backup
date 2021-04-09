#%%
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from python_speech_features import mfcc
from python_speech_features import logfbank
from preprocessing_for_data import new_minmax_normal


def detect_wav_file(data):
    rate_max = 0.2

    temp_max = np.max(data)
    print(temp_max)
    for i,j in enumerate(data):
        if temp_max*rate_max < j:
            start_index = i
            break

    for i,j in enumerate(reversed(data)):
        if temp_max*rate_max < j:
            end_index = i
            break

    print(start_index, end_index)
    result = data[start_index:end_index]

    add_data = np.random.rand(3000)*10

    temp = np.append(add_data, result)
    result = np.append(temp, add_data)

    return result



def main():

    # start_point = 10000

    fs_0, data_0 = wavfile.read(wav_path7)
    fs_1, data_1 = wavfile.read(wav_path8)

    plt.subplot(4,1,1)
    plt.plot(data_0)

    plt.subplot(4,1,2)
    plt.plot(data_1)

    # data_0 = detect_wav_file(data_0)

    # data_0 = data_0[start_point:start_point+15000]

    data_0 = logfbank(data_0, fs_0)
    data_1 = new_minmax_normal(data_0)

    data_1 = logfbank(data_1, fs_1)
    data_1 = new_minmax_normal(data_1)

    plt.subplot(4,1,3)
    plt.pcolormesh(data_0)
    # plt.plot(data_0)
    plt.subplot(4,1,4)
    plt.pcolormesh(data_1)
    # plt.plot(data_1)

    plt.show()

    return



if __name__ == '__main__':
    main()
