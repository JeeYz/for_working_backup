import os
import json

from scipy.io import wavfile
import numpy as np


file_ext = ".wav"
parent_folder = "D:\\voice_data_backup\\PNC_DB_ALL"

command_list = [

        "camera", "end", "picture",
        "stop", "record"

    ]





def initialization_of_dict(data_dict):

    data_dict["title"] = "whole data for training"

    data_dict["speakers"] = list()

    speakers_list = list()

    for (path, dir, files) in os.walk(parent_folder):

        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                speaker = path.split("\\")[-2]

                if speaker not in speakers_list:
                    # 화자가 리스트에 없을 때
                    speakers_list.append(speaker)
                                    
    # print(len(speakers_list))

    for one_speaker in speakers_list:
        # 한 명의 화자당
        temp = dict()
        temp["name"] = one_speaker
        temp["commands"] = list()

        for one_command in command_list:
            tmp_com = dict()
            tmp_com["command"] = one_command
            tmp_com["saturation_data"] = list()
            tmp_com["normal_data"] = list()
            temp["commands"].append(tmp_com)
        
        data_dict["speakers"].append(temp)
        

    return data_dict 




def determine_condition(file_name):

    try:
        sample_rate, data = wavfile.read(file_name)
    except wavfile.WavFileWarning:
        print("\noccur warning...\n")

    max_value = np.max(data)
    min_value = np.min(data)
    sorted_np, indices = np.unique(data, return_counts=True)

    # 조건 0 and 1
    if max_value != 32767   \
                or         \
            min_value != -32768:                        

        return 1
    else:
        return 0

    # correct_data_num+=1

    # 조건 2 and 3
    # if max_value == 32767       \
    #             and             \
    #         indices[-1] > 10000:
    #     return 0
    # elif min_value == -32768       \
    #             and             \
    #         indices[0] > 10000:
    #     return 0
    # else:
    #     return 1
    




def insert_to_dict(whole_data, file_name):

    temp_fn = file_name.split('\\')

    for one_speaker in whole_data["speakers"]:

        if temp_fn[-3] == one_speaker["name"]:
            for one_command in one_speaker["commands"]:

                if temp_fn[-2] == one_command["command"]:
                    cond = determine_condition(file_name)
                    
                    if cond == 0:
                        one_command["saturation_data"].append(file_name)
                    elif cond == 1:
                        one_command["normal_data"].append(file_name)

    return whole_data




def search_wav_files(whole_data):

    for (path, dir, files) in os.walk(parent_folder):

        for filename in files:

            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                temp_split = path.split("\\")
                if temp_split[-1] in command_list:
                    file_name = path + "\\" + filename

                    whole_data = insert_to_dict(whole_data, file_name)

    return whole_data




def write_json_file(whole_data):

    json_file = "or_condition_data.json"
    with open(json_file, 'w', encoding='utf-8') as jf:
        json.dump(whole_data, jf, indent=4)

    return




def main():

    whole_data = dict()
    whole_data = initialization_of_dict(whole_data)
    whole_data = search_wav_files(whole_data)
    write_json_file(whole_data)

    return











if __name__ == "__main__":
    print('hello, world~!!')
    main()
