import sys
sys.path.append('D:\\')
sys.path.append('..\\')

from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn import preprocessing

from python_speech_features import logfbank
from files_locations import example_filename
from draw_single_graph import draw_graph_logfbank, draw_graph_logfbank_norm, draw_graph_raw_signal
import draw_single_graph
import draw_multi_graphs

from util_module import standardization_func, new_minmax_normal, transpose_the_matrix
import signal_trigger
import find_wav_files

trigger_val = 0.5


##
def modifying_train_data():



    return


## main func
def main():

    recording_time = 2

    sr, data = wavfile.read(example_filename)
    data = standardization_func(data)

    data = signal_trigger.evaluate_mean_of_frame(data, frame_time=0.025, shift_time=0.01, sample_rate=sr)
    draw_graph_raw_signal(data, title_name='Cut Raw Signal')
    plt.show()

    return













## main func
if __name__=='__main__':
    main()
