
import os
from scipy.io import wavfile
import numpy as np
import json

text_wav_files_dict = dict()
delete_keywords_list = list()

text_file_path = ['D:\\voice_data_backup\\kr_label', 'D:\\voice_data_backup\\hk_label']
wav_file_path = 'D:\\voice_data_backup\\zeroth_korean.tar\\zeroth_korean\\train_data_01\\003'


labels_list_0 = [
                    '카메라', '촬영', '녹화', '중지', '종료' 
]

labels_list_1 = [
                    '선택', '클릭', '닫기', '홈', '종료', '어둡게', '밝게', '음성명령어', 
]

labels_list_2 = [
                    '촬영', '녹화', '정지', 
]

labels_list_3 = [
                    '위로', '아래로', '다음', '이전', 
]

labels_list_4 = [
                    '재생', '되감기', '빨리감기', '처음', '소리작게', '소리크게',                    
]

labels_list_5 = [
                    '화면크게', '화면작게', '전체화면', '이동', '멈춤', '모든창보기', 
]

labels_list_6 = [
                    '전화', '통화', '수락', '거절', 
]

labels_list_7 = [
                    '화면 크게', '화면 작게', '전체 화면', 
                    '소리 작게', '소리 크게', 
                    '음성 명령어',
]

all_labels = [
                labels_list_0,
                labels_list_1,
                labels_list_2,
                labels_list_3,
                labels_list_4,
                labels_list_5,
                labels_list_6,
            ]

#%%
def check_labels_text():

    sent_num = 0

    print(len(text_wav_files_dict.keys()))
    for one_key in text_wav_files_dict.keys():
        temp = text_wav_files_dict[one_key]['sentence']
        
        for one_label in all_labels:
            for one_command in one_label:
                if one_command in temp:
                    # print(temp, one_key, one_command)
                    if one_key not in delete_keywords_list:
                        delete_keywords_list.append(one_key)
                    sent_num+=1
    print(sent_num)
    return 




#%%
def delete_keyword():

    for one in delete_keywords_list:
        del text_wav_files_dict[one]
    
    print(len(text_wav_files_dict.keys()))

    return



#%%
def add_time_tables():

    for one_key in text_wav_files_dict.keys():
        
        file_temp = text_wav_files_dict[one_key]['txt']
        with open(file_temp, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                line = line.split()
                if not line:
                    break
                
                try:
                    sr, signal_data = wavfile.read(one_key)
                except FileNotFoundError:
                    continue

                temp = dict()

                temp['start'] = int(sr*float(line[0]))
                start_point = temp['start']
                temp['end'] = int(sr*float(line[1]))
                end_point = temp['end']
                temp['data'] = signal_data[start_point:end_point]

                text_wav_files_dict[one_key]['content'].append(temp)


    return



#%%
def read_sentence(para_path, para_name):
    
    temp_filename = para_path.split("\\")[-1]
    temp_path = para_path + "\\" + temp_filename + "_003.trans.txt"

    with open(temp_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            line = line.split()
            if not line:
                break
            if line[0] == para_name:
                temp_line = ' '.join(line[1:])
                return temp_line




#%%
def gen_text_wav_files_dict():

    file_ext = ".txt"

    for one_path in text_file_path:
        for (path, dir, files) in os.walk(one_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == file_ext:
                    file_name = path + "\\" + filename
                    temppath = wav_file_path + "\\" + filename.split("_")[0]
                    tempname = filename.split(".")[-2]
                    temp = temppath + "\\" + tempname + ".wav"

                    if not os.path.exists(temp):
                        temp_wav_path = 'D:\\voice_data_backup\\zeroth_korean.tar\\zeroth_korean\\test_data_01\\003'
                        temppath = temp_wav_path + "\\" + filename.split("_")[0]
                        tempname = filename.split(".")[-2]
                        temp = temppath + "\\" + tempname + ".wav"

                    if os.path.exists(temp):
                        tempsent = read_sentence(temppath, tempname)
                        text_wav_files_dict[temp] = dict()
                        text_wav_files_dict[temp]['txt'] = file_name
                        text_wav_files_dict[temp]['sentence'] = tempsent
                        text_wav_files_dict[temp]['content'] = list()                    
                    else:
                        continue
                    
    


#%%
def print_contents_of_dict():

    print(json.dumps(text_wav_files_dict, sort_keys=False, indent=4, default=str, ensure_ascii=False))

    return




#%%
def main():

    gen_text_wav_files_dict()

    # print_contents_of_dict()
    add_time_tables()
    print_contents_of_dict()
    check_labels_text()
    # print_contents_of_dict()
    delete_keyword()

    return






#%%
if __name__ == "__main__":
    print('hello, world~!!')
    main()






