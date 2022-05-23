import os
import shutil
from pickle import FALSE
import sys
from scipy.io import wavfile
import wave
from enum import Enum, auto

import numpy as np
import json
import librosa
import wavio
import scipy.signal as sps

import time
import copy
import matplotlib.pyplot as plt

import tensorflow as tf
import threading
import pyaudio as pa

import augment_processing as augp
import signal_processing as sigp
import trigger_algorithm as triga
import files_module as fm


# full size
FULL_SIZE = 40000
RAW_DATA_FULLSIZE = 64000

# start index
START_INDEX_THRESHOLD = 10000

# train data types
TRAIN_DATA_TYPE = np.float32

# boolean variables
GEN_TRAIN_INCLUDE_NONE = False
INCLUDE_ZEROTH_NONE =  True

GEN_DATA_TYPE = 'train'

PREPRO_SHIFT_SIZE = 200
PREPRO_FRAME_SIZE = 400

MAX_SIGNAL_VALUE = 32767.0
MIN_SIGNAL_VALUE = -32768.0

GLOBAL_THRESHOLD_VALUE = 1000
GLOBAL_THRESHOLD = 0.5 

rate_list = [
    # 0.95,
    1.0,
    1.05,
    1.10,
    # 1.15,
]


NOISE_RATIO = [
    0.5,
    1.0,
    1.5,
    2.0,
]


# data pathes
CWdata_path = '/home/pncdl/DeepLearning/CWtraindata/PnC_Solution_CW_all_1102/'
CWdata_16k_path = '/home/pncdl/DeepLearning/CWtraindata/PnC_Solution_CW_all_1102_addnoise/'
npz_target_path = '/home/pncdl/DeepLearning/CWtraindata/npzTrain/'

whole_data_json_filename = '$$whole_data_info.json'
whole_data_json_filename_16k = '$$whole_data_info_16k.json'
target_numpy_dir_name = "npz_train"

noise_data_path = '/home/pncdl/DeepLearning/CWtraindata/noise_data/'


# check traindata variables
NUMBER_OF_GRAPH = 49


# Stereo 24bit 48,000Hz
LABEL_BLOCK_SIZE = 5
TARGET_SAMPLE_RATE = 16000
RESOURCE_SAMPLE_RATE = 48000
RESOURCE_SECS = 4
RESOURCE_FULL_SIZE = RESOURCE_SAMPLE_RATE*RESOURCE_SECS

AUGMENT_FLAG = 1


# speaker exception
speaker_exception = {
    'strong':[
        '59185653',
        '59184297',
        '59184290',
        '59185228',
    ],
    'weak':[
        '59183640',
        '59184387',
        '59185145',
        '59185238',
        '59185713',
        '59184211',
        '59192927',
    ],
    'machine_noise':[
        '59184088',
        '59185312',
        '59185353',
        '59185622',
        '59185031',
        '59184205',
        '59184752',
    ],
    'none':[
        '59185990',
        '59198984',
    ],
}


class LabelsKorEng(Enum):
    # from 1~ <- 요 순서대로 정렬되어 있음

    CHOICE=auto()               # 선택  1
    CLICK=auto()                # 클릭  2
    CLOSE=auto()                # 닫기  3 
    HOME=auto()                 # 홈    4
    END=auto()                  # 종료  5
    DARKEN=auto()               # 어둡게    6
    BRIGHTEN=auto()             # 밝게  7
    VOICE_COMMANDS=auto()       # 음성 명령어   8
    PICTURE=auto()              # 촬영  9
    RECORD=auto()               # 녹화  10
    STOP=auto()                 # 정지  11
    DOWN=auto()                 # 아래로    12
    UP=auto()                   # 위로  13
    NEXT=auto()                 # 다음  14
    PREVIOUS=auto()             # 이전  15
    PLAY=auto()                 # 재생  16
    REWIND=auto()               # 되감기    17
    FAST_FORWARD=auto()         # 빨리감기  18
    INITIAL_POSITION=auto()     # 처음  19
    VOLUME_DOWN=auto()          # 소리 작게 20
    VOLUME_UP=auto()            # 소리 크게 21
    BIG_SCREEN=auto()           # 화면 크게 22
    SMALL_SCREEN=auto()         # 화면 작게 23
    FULL_SCREEN=auto()          # 전체 화면 24
    MOVE=auto()                 # 이동  25
    FREEZE=auto()               # 멈춤  26
    SHOW_ALL_WINDOWS=auto()     # 모든 창 보기  27
    PHONE=auto()                # 전화  28
    CALL=auto()                 # 통화  29
    ACCEPT=auto()               # 수락  30
    REJECT=auto()               # 거절  31









if __name__ == '__main__':
    print(LabelsKorEng.CHOICE.value)
    for i in LabelsKorEng:
        print(i, i.value, type(i.value), i.name, type(i.name))

    # print(LabelsKorEng.name[1])





