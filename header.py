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
