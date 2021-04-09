import sys
sys.path.append('D:\\')
temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile

from python_speech_features import mfcc
from python_speech_features import logfbank

import draw_single_graph
from util_module import standardization_func, new_minmax_normal, transpose_the_matrix

titles_list = []
xlabels_list = []
ylabels_list = []

def compare_data(*args, **kwargs):

    if "data_dict" in kwargs.keys():
        data_dict = kwargs['data_dict']
    if "x_number" in kwargs.keys():
        x_num = kwargs['x_number']
    if "y_number" in kwargs.keys():
        y_num = kwargs['y_number']

    ## draw signal graph
    plt.figure()
    plt.suptitle("Signal Graph", fontsize=16)
    for i in range(1, data_dict['num_data']+1):
        temp = data_dict['data'][i-1]
        plt.subplot(x_num, y_num, i)
        plt.plot(temp[0])
        plt.title(temp[1])
        plt.xlabel(temp[2])
        plt.ylabel(temp[3])
        # plt.tight_layout()

    ## draw feature vector graph
    plt.figure()
    plt.suptitle("Feature Vector Graph", fontsize=16)
    for i in range(1, data_dict['num_data']+1):
        temp = data_dict['data'][i-1]
        data = logfbank(temp[0], data_dict['sample_rate'])
        data = draw_single_graph.transpose_the_matrix(data)
        plt.subplot(x_num, y_num, i)
        plt.pcolormesh(data)
        plt.title(temp[1])
        plt.xlabel(temp[2])
        plt.ylabel(temp[3])
        # plt.tight_layout()
        plt.colorbar()

    ## draw normalized feature vector graph
    plt.figure()
    plt.suptitle("Normalized Feature Vector Graph", fontsize=16)
    for i in range(1, data_dict['num_data']+1):
        temp = data_dict['data'][i-1]
        data = logfbank(temp[0], data_dict['sample_rate'])
        data = draw_single_graph.new_minmax_normal([data])
        data = data[0]
        data = draw_single_graph.transpose_the_matrix(data)
        plt.subplot(x_num, y_num, i)
        plt.pcolormesh(data)
        plt.title(temp[1])
        plt.xlabel(temp[2])
        plt.ylabel(temp[3])
        # plt.tight_layout()
        plt.colorbar()


    return


def draw_multiple_graph_with_one_data(data, sr, **kwargs):

    if "x_number" in kwargs.keys():
        x_num = kwargs['x_number']
    if "y_number" in kwargs.keys():
        y_num = kwargs['y_number']

    plt.subplot(x_num, y_num, 1)
    plt.plot(data)
    plt.title('Raw Signal')
    plt.xlabel('sample rate')
    plt.ylabel('amplitude')


    data = logfbank(data, sr)
    data = draw_single_graph.transpose_the_matrix(data)
    plt.subplot(x_num, y_num, 2)
    plt.pcolormesh(data)
    plt.title('Feature Vector')
    plt.xlabel('frame sequence')
    plt.ylabel('number of filters')
    plt.colorbar()

    data = draw_single_graph.new_minmax_normal([data])
    data = data[0]
    plt.subplot(x_num, y_num, 3)
    plt.pcolormesh(data)
    plt.title('Normalized Feature Vector')
    plt.xlabel('frame sequence')
    plt.ylabel('number of filters')
    plt.colorbar()

    return


def compare_data_in_one(**kwargs):

    if "data_dict" in kwargs.keys():
        data_dict = kwargs['data_dict']
    if "x_number" in kwargs.keys():
        x_num = kwargs['x_number']
    if "y_number" in kwargs.keys():
        y_num = kwargs['y_number']

    sr = data_dict['sample_rate']
    sig_data = data_dict['data']

    ## raw signal
    plt.figure()
    plt.suptitle("Raw Signal Graph", fontsize=16)
    draw_multiple_graph_with_one_data(sig_data[0][0], sr, x_number=3, y_number=1)

    ## std raw signal
    plt.figure()
    plt.suptitle("Std Raw Signal Graph", fontsize=16)
    draw_multiple_graph_with_one_data(sig_data[1][0], sr, x_number=3, y_number=1)

    ## windowing raw signal
    plt.figure()
    plt.suptitle("Windowing Raw Signal Graph", fontsize=16)
    draw_multiple_graph_with_one_data(sig_data[2][0], sr, x_number=3, y_number=1)

    ## std windowing raw signal
    plt.figure()
    plt.suptitle("Std Windowing Raw Signal", fontsize=16)
    draw_multiple_graph_with_one_data(sig_data[3][0], sr, x_number=3, y_number=1)

    return













## endl
