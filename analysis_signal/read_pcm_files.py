

import os

import matplotlib.pyplot as plt
from numpy.lib.npyio import load
from scipy.io import wavfile

import numpy as np
import json


pcm_path = 'D:\\voice_data_backup\\AI_HUB_data_speech\\'
pcm_text_list = 'D:\\voice_data_backup'



def read_one_pcm_file_for_test():

    global pcm_path

    file_path = "KsponSpeech_01\\KsponSpeech_0001\\KsponSpeech_000001.pcm"

    one_pcm_file_path = pcm_path + file_path

    # sr, data = wavfile.read(one_pcm_file_path, "rb")

    with open(one_pcm_file_path, 'rb') as rf:

        buf = rf.read()

        # print(len(buf))

        print(type(buf))

        buf = buf.split()

        print(type(buf))

        for i in range(20):
            print(buf[i])


        
        # data = np.frombuffer(buf, dtype=np.int16)

        # print(data)

        # print(len(data))

        # plt.plot(data)

        # plt.show()


    return




def read_json_wav_data_for_none():

    file_path = "D:\\voice_data_backup\\명령어 음성(일반남녀) 데이터\\Training\\3.키오스크_라벨링_명령어(일반)_training\\kiosk\\script1_j_0040\\script1_j_0040-9001-01-03-HDB-F-05-A.json"

    with open(file_path, "rb") as jf:
        loaded_data = json.load(jf)

    print(json.dumps(loaded_data, indent=4, ensure_ascii=False))
    # print(loaded_data)

    return




def find_json_files():

    parent_path = "D:\\voice_data_backup\\명령어 음성(일반남녀) 데이터\\Training\\"

    files_num = 0

    for (path, dir, files) in os.walk(parent_path):
        for filename in files:

            ext = os.path.splitext(filename)[-1]
            if ext == ".wav":
                files_num +=1

    print(files_num)

    return




##
def main():

    # read_one_pcm_file_for_test()

    # read_json_wav_data_for_none()
    find_json_files()

    return





if __name__ == '__main__':
    main()

## endl
