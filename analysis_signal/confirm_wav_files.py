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
from analysis_signal.util_module import standardization_func, new_minmax_normal, transpose_the_matrix


wav_dict_path = "D:\\voice_data_backup\\W20_voice_data"


##
def make_files_list():

    hipnc_list = list()
    camera_list = list()
    end_list = list()
    init_list = list()
    position_list = list()
    stop_list = list()

    for (path, dir, files) in os.walk(wav_dict_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.wav':
                temp_name = filename.split('_')
                temp_name = temp_name[1]
                if 'hipnc' in temp_name:
                    hipnc_list.append(path+'\\'+filename)

                elif 'camera' in temp_name:
                    camera_list.append(path+'\\'+filename)

                elif 'end' in temp_name:
                    end_list.append(path+'\\'+filename)

                elif 'init' in temp_name:
                    init_list.append(path+'\\'+filename)

                elif 'position' in temp_name:
                    position_list.append(path+'\\'+filename)

                elif 'stop' in temp_name:
                    stop_list.append(path+'\\'+filename)

                # fs, data = wavfile.read(path+'\\'+filename)

    return hipnc_list, camera_list, end_list, init_list, position_list, stop_list












##
if __name__ == '__main__':
    hipnc_list, camera_list, end_list, init_list, position_list, stop_list = make_files_list()
    full_data_list = [
                        hipnc_list,
                        camera_list,
                        end_list,
                        init_list,
                        position_list,
                        stop_list
                    ]

    for i in full_data_list:
        for j in i:

            fs, data = wavfile.read(j)

            plt.figure()
            plt.plot(data)

            plt.xlabel('sample rate')
            plt.ylabel('amplitude')
            plt.title(j)

    plt.show()





## endl
