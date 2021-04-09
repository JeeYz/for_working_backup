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

data_dict = dict()


## draw single graphs
def draw_single_graphs():

    temp_data = data_dict['data']
    sr = data_dict['sample_rate']

    draw_graph_raw_signal(temp_data[0][0], title_name='raw signal')
    draw_graph_logfbank(temp_data[0][0], sr, title_name='logfbank')
    draw_graph_logfbank_norm(temp_data[0][0], sr, title_name='normalized logfbank')

    draw_graph_raw_signal(temp_data[1][0], title_name='std raw signal')
    draw_graph_logfbank(temp_data[1][0], sr, title_name='std logfbank')
    draw_graph_logfbank_norm(temp_data[1][0], sr, title_name='std normalized logfbank')

    draw_graph_raw_signal(temp_data[2][0], title_name='windowing raw signal')
    draw_graph_logfbank(temp_data[2][0], sr, title_name='windowing logfbank')
    draw_graph_logfbank_norm(temp_data[2][0], sr, title_name='windowing normalized logfbank')

    draw_graph_raw_signal(temp_data[3][0], title_name='windowing raw signal std')
    draw_graph_logfbank(temp_data[3][0], sr, title_name='windowing logfbank std')
    draw_graph_logfbank_norm(temp_data[3][0], sr, title_name='windowing normalized logfbank std')

    return


## main func
def main():

    ## raw data
    sr, data = wavfile.read(example_filename)
    data_dict['sample_rate'] = sr
    raw_sig = [data, 'raw signal', 'sample rate', 'amplitude']

    ## standardization
    std_data = draw_single_graph.standardization_func(data)
    std_raw_sig = [std_data, 'std raw signal', 'frame sequence', 'number of filters']

    ## windowing with raw data
    window = signal.windows.hann(len(data))
    win_data = data*window
    win_sig = [win_data, 'windowing raw signal', 'frame sequence', 'number of filters']

    ## windowing with stdardized data
    window = signal.windows.hann(len(std_data))
    win_std_data = std_data*window
    win_sig_std = [win_std_data, 'windowing raw signal std', 'frame sequence', 'number of filters']


    data_dict['data'] = [raw_sig, std_raw_sig, win_sig, win_sig_std]
    data_dict['num_data'] = len(data_dict['data'])

    # draw_single_graphs()
    draw_multi_graphs.compare_data(data_dict=data_dict, x_number=2, y_number=2)
    draw_multi_graphs.compare_data_in_one(data_dict=data_dict, x_number=3, y_number=1)

    plt.show()

    return













if __name__ == '__main__':
    main()

















## endl
