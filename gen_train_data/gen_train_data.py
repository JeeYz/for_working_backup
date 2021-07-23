import numpy as np
import json
import os



command_dict = {    
                "camera": 0,
                "picture": 1,
                "record": 2, 
                "stop": 3,
                "end": 4,                
                }

none_label_num = 5

def make_all_files_list():

    all_files = list()
    parent_folder = "D:\\voice_data_backup\\PNC_DB_ALL"
    file_ext = '.wav'

    for (path, dir, files) in os.walk(parent_folder):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                file_name = path + "\\" + filename
                all_files.append(file_name)

    return all_files




def generate_train_data(json_data, null_list, all_files):

    npz_file = "D:\\train_data_satu_or_condition.npz"

    train_data = list()
    label_list = list()

    for one_file in all_files:
        for one_command in json_data.keys():

            

            pass
        
    train_data = np.array(train_data, dtype=np.float32)
    label_list = np.array(label_list, dtype=np.int16)

    np.savez()

    return




def initialize_dict(data_dict):
    
    for one in command_dict.keys():
        data_dict[one] = list()
    
    return data_dict




def read_json_file():

    new_dict = dict()
    new_dict = initialize_dict(new_dict)

    null_list = list()

    json_file = "or_condition_data.json"
    with open(json_file, 'r', encoding='utf-8') as jf:
        json_data = json.load(jf)

    for one_speaker in json_data["speakers"]:
        for one_command in one_speaker["commands"]:
            if one_command["normal_data"] != []:
                for npath in one_command["normal_data"]:
                    new_dict[one_command["command"]].append(npath)
                
            if one_command["saturation_data"] != []:
                for npath in one_command["saturation_data"]:
                    null_list.append(npath)

    return new_dict, null_list





def main():

    json_data, null_list = read_json_file()
    all_files = make_all_files_list()
    generate_train_data(json_data, null_list, all_files)

    return





if __name__ == '__main__':
    print("hello, world~!!")
    main()