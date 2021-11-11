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
from CW_class_data import TrainData, DecodingData

import tensorflow as tf
import threading
import pyaudio as pa


# global import modules
import file_processing as fpro
import signal_processing as spro
import test_and_check as tcheck
import augment_processing as augp
import trigger_algorithm as trigal
import modifying_data_and_info as moddi
import none_aug_processing as nonepro
import gen_data_files as gendata
import CW_json_files_decoder as decoder
import CW_signal_processing as cwsig



# full size
FULL_SIZE = 40000

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
GLOBAL_THRESHOLD_TEST = 1.5


# boolean variables
GEN_TRAIN_INCLUDE_NONE = False
INCLUDE_ZEROTH_NONE =  True

# variables for generating data 
POP_DATA_NUM = 3
DATA_AUG_POSITION = 10

PREPRO_SHIFT_SIZE = 200
PREPRO_FRAME_SIZE = 400

HEAD_SIZE = 7000
TAIL_SIZE = 3000

BUFFER_SIZE = 8000

BLOCK_OF_RANDOM = 500

MAX_SIGNAL_VALUE = 32767.0
MIN_SIGNAL_VALUE = -32768.0


rate_list = [
                1.10,
                # 1.0,
                # 1.05,
                # 1.07,

    ]

# data pathes
numpy_traindata_files_path = 'D:\\GEN_train_data_Ver.1.0.npz'
numpy_traindata_files_path_zero = 'D:\\GEN_train_data_Ver_zero_pad.1.0.npz'
numpy_testdata_files_path = 'D:\\GEN_train_data_Ver.1.0_test_.npz'

numpy_traindata_file_CWdata = "D:\\GEN_train_data_Ver.1.0_CWdata.npz"
# json_file_CWdata = "D:\\json_train_data_Ver.1.0.json"
json_file_CWdata = "C:\\temp\\json_train_data_Ver.1.0.json"
# NUMPY_TRAINDATA_FILE_CWDATA = "D:\\GEN_train_data_Ver.1.0_CWdata.npz"
# JSON_FILE_CWDATA = "D:\\json_train_data_Ver.1.0.json"

none_data_path = "D:\\voice_data_backup\\zeroth_none_label"
train_data_path = "D:\\voice_data_backup\\PNC_DB_ALL"
test_data_path = "D:\\voice_data_backup\\test"

# CWdata_path = 'D:\\voice_data_backup\\CW_voice_data'
CWdata_path = 'C:\\temp'
npz_target_path = 'C:\\temp\\npz_train\\'

tflite_file = "D:\\PNC_ASR_2.4_CW_model_.tflite"


# check traindata variables
NUMBER_OF_GRAPH = 20


# Stereo 24bit 48,000Hz
LABEL_BLOCK_SIZE = 5
TARGET_SAMPLE_RATE = 16000
RESOURCE_SAMPLE_RATE = 48000
RESOURCE_SECS = 4
RESOURCE_FULL_SIZE = RESOURCE_SAMPLE_RATE*RESOURCE_SECS

AUGMENT_FLAG = 1

# decoder audio constants
CHUNK_SIZE = 400
RECORDING_TIME = 3
PER_SEC = TARGET_SAMPLE_RATE/CHUNK_SIZE
RETURN_STACK_SIZE = (TARGET_SAMPLE_RATE/CHUNK_SIZE)*(RECORDING_TIME)

NUM_CHANNEL = 1

VOICE_TRIGGER = 200
STD_TRIGGER = 1.0
NORM_TRIGGER = 0.001

CMD_LABEL_NUM = 32


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
    UP=auto()                   # 위로  12
    DOWN=auto()                 # 아래로    13
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

# declare global class
# GLOBAL_CW_TRAINDATA = TrainData(
#     dtype=TRAIN_DATA_TYPE,
#     json_path=json_file_CWdata,
#     numpy_path=numpy_traindata_file_CWdata,
# )

GLOBAL_DECODING_DATA = DecodingData()



if __name__ == '__main__':
    print(LabelsKorEng.CHOICE.value)
    for i in LabelsKorEng:
        print(i, i.value, type(i.value), i.name, type(i.name))

    # print(LabelsKorEng.name[1])





