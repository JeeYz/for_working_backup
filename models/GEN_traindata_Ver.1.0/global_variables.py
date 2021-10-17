import os
import sys
from scipy.io import wavfile
import wave

import numpy as np
import json
import librosa
import wavio
import scipy.signal as sps

import time
import copy
import matplotlib.pyplot as plt

# full size
FULL_SIZE = 20000

# train data types
TRAIN_DATA_TYPE = np.float32

# variables for experiments
TIME_STRETCH_NUM = 2
X_AXIS_SIZE = 500
GLOBAL_THRESHOLD_RATE = 0.1
GLOBAL_THRESHOLD_RATE_TEST = 0.4
GLOBAL_THRESHOLD_VALUE = 1000
GLOBAL_NOISE_THRESHOLD_RATE = 0.001

GLOBAL_THRESHOLD = 1.0 
GLOBAL_THRESHOLD_TEST = 1.0

# boolean variables
GEN_TRAIN_INCLUDE_NONE = False

# variables for generating data 
POP_DATA_NUM = 3
DATA_AUG_POSITION = 10

PREPRO_SHIFT_SIZE = 200
PREPRO_FRAME_SIZE = 400

HEAD_SIZE = 2000
TAIL_SIZE = 2000

NORM_STAN_PARA = 'stan'

BLOCK_OF_RANDOM = 100

MAX_SIGNAL_VALUE = 32767.0
MIN_SIGNAL_VALUE = -32768.0

# label dictionary
label_dict = {
        'camera' : 1,
        'picture' : 2,
        'record' : 3,
        'stop' : 4,
        'end' : 5,
        'none' : 0,
    }

rate_list = [
                1.10,
                # 1.0,
                1.05,

    ]


# data pathes
numpy_traindata_files_path = 'D:\\GEN_train_data_Ver.1.0.npz'
numpy_testdata_files_path = 'D:\\GEN_train_data_Ver.1.0_test_.npz'

none_data_path = "D:\\voice_data_backup\\zeroth_none_label"
train_data_path = "D:\\voice_data_backup\\PNC_DB_ALL"
test_data_path = "D:\\voice_data_backup\\test"

