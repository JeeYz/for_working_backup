
import os
import numpy as np
import wave
from scipy.io import wavfile

pcm_file_path = "D:\\voice_data_backup\\AI_HUB_data_speech\\"

wav_file_path = "D:\\voice_data_backup\\ASR_audio_files\\"

zeroth_file_path = "D:\\voice_data_backup\\zeroth_korean.tar\\zeroth_korean\\"


def find_wav_files(fpath, file_format, p_sr):

    count = 0
    sum_sr = 0

    for (path, dir, files) in os.walk(fpath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_format:
                count += 1
                fs, data = wavfile.read(path+'\\'+filename)
                sum_sr += len(data)/p_sr

    return count, sum_sr

def find_pcm_files(fpath, file_format, p_sr):

    count = 0
    sum_sr = 0

    for (path, dir, files) in os.walk(fpath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_format:
                count += 1
                data = np.memmap(path+'\\'+filename, dtype='h', mode='r')
                sum_sr += len(data)/p_sr
                print('\rone file is done....\t%d' %count, end='')

    return count, sum_sr



def main():

    # temp_c, temp_sr = find_wav_files(wav_file_path, '.wav', 16000)
    # print(temp_c, temp_sr/3600)
    #
    # temp_c, temp_sr = find_wav_files(zeroth_file_path, '.wav', 16000)
    # print(temp_c, temp_sr/3600)

    temp_c, temp_sr = find_pcm_files(pcm_file_path, '.pcm', 44100)
    print(temp_c, temp_sr/3600)

    return








##
if __name__ == '__main__':
    main()



## endl
