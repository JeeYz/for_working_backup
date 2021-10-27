import os
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
from CW_class_data import TrainData

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
# FULL_SIZE = 20000
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

HEAD_SIZE = 5000
TAIL_SIZE = 3000

BUFFER_SIZE = 5000

BLOCK_OF_RANDOM = 500

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


class Labels(Enum):
    CAMERA = 1
    PICTURE = 2 
    RECORD = 3 
    STOP = 4
    END = 5
    NONE = 0


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
json_file_CWdata = "D:\\json_train_data_Ver.1.0.json"
# NUMPY_TRAINDATA_FILE_CWDATA = "D:\\GEN_train_data_Ver.1.0_CWdata.npz"
# JSON_FILE_CWDATA = "D:\\json_train_data_Ver.1.0.json"

none_data_path = "D:\\voice_data_backup\\zeroth_none_label"
train_data_path = "D:\\voice_data_backup\\PNC_DB_ALL"
test_data_path = "D:\\voice_data_backup\\test"

CWdata_path = 'D:\\voice_data_backup\\CW_voice_data'

tflite_file = "D:\\PNC_ASR_2.4_CW_data_.tflite"


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
RECORDING_TIME = 4


class LabelsKorEng(Enum):
    # from 1~ <- 요 순서대로 정렬되어 있음

    CHOICE=auto()               # 선택
    CLICK=auto()                # 클릭
    CLOSE=auto()                # 닫기
    HOME=auto()                 # 홈
    END=auto()                  # 종료
    DARKEN=auto()               # 어둡게
    BRIGHTEN=auto()             # 밝게
    VOICE_COMMANDS=auto()       # 음성 명령어
    PICTURE=auto()              # 촬영
    RECORD=auto()               # 녹화
    STOP=auto()                 # 정지
    UP=auto()                   # 위로
    DOWN=auto()                 # 아래로
    NEXT=auto()                 # 다음
    PREVIOUS=auto()             # 이전
    PLAY=auto()                 # 재생
    REWIND=auto()               # 되감기
    FAST_FORWARD=auto()         # 빨리감기
    INITIAL_POSITION=auto()     # 처음
    VOLUME_DOWN=auto()          # 소리 작게
    VOLUME_UP=auto()            # 소리 크게
    BIG_SCREEN=auto()           # 화면 크게
    SMALL_SCREEN=auto()         # 화면 작게
    FULL_SCREEN=auto()          # 전체 화면
    MOVE=auto()                 # 이동
    FREEZE=auto()               # 멈춤
    SHOW_ALL_WINDOWS=auto()     # 모든 창 보기
    PHONE=auto()                # 전화
    CALL=auto()                 # 통화
    ACCEPT=auto()               # 수락
    REJECT=auto()               # 거절

# declare global class
GLOBAL_CW_TRAINDATA = TrainData(
    dtype=TRAIN_DATA_TYPE,
    json_path=json_file_CWdata,
    numpy_path=numpy_traindata_file_CWdata,
)



if __name__ == '__main__':
    print(LabelsKorEng.CHOICE.value)
    for i in LabelsKorEng:
        print(i, i.value, type(i.value), i.name, type(i.name))

    # print(LabelsKorEng.name[1])





