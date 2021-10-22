
from pickle import NONE
from global_variables import *
import file_processing as fpro


def CW_load_json_files():
    files_list = fpro.find_data_files(CWdata_path, '.json')

    whole_json_data = list()

    for one_jsonfile in files_list:
        print(one_jsonfile)
        with open(one_jsonfile, 'r', encoding='utf-8') as jf:
            loaded_jsondata = json.load(jf)
        whole_json_data.append(loaded_jsondata)

    return whole_json_data


def refine_loaded_data(whole_json_data):
    temp_list = list()

    for one_jsonfile in whole_json_data:
        temp_result = one_jsonfile['result']
        
        for one_result in temp_result:
            for one_key in one_result.keys():
                if "name_" in one_key:
                    target_key = one_key
            
                    data_dict_list = one_result[target_key]['data']
                    result_list = gen_label_data(
                        data_dict_list, 
                        one_result['dataID']
                    )

                    temp_list.extend(result_list) 

    return temp_list


def gen_label_data(data_dict_list, speaker_name):
    result_list = list()
    find_class = FindFilePath()

    for one_list in data_dict_list:
        data_dict = one_list['value']
        gl_class = GenLabels()

        if len(data_dict) is 1:
            non_cmd_flag = True 
        else:
            non_cmd_flag = False

        for i, one_filename in enumerate(data_dict):
            temp_dict = dict()
            temp = one_filename['file_name']
            temp_dict['speaker'] = speaker_name
            temp_dict['filename'] = temp

            if non_cmd_flag is True:
                temp_dict['label'] = 0
            else:
                temp_dict['label'] = gl_class.get_label(i)

            return_dict = find_class.get_filepath(temp_dict) 
            result_list.append(return_dict)
        
        print(i+1)

    return result_list 


class FindFilePath():
    def __init__(self):
        self.__data_path = CWdata_path
        self.__cmd_files_path = self.set_cmd_path()
        self.__nonecmd_files_path = self.set_nonecmd_path()

    def get_filepath(self, one_data_dict):
        temp_filename = one_data_dict['filename']

        if one_data_dict['label'] is 0:
            temp = self.__nonecmd_files_path+'\\'+temp_filename
            one_data_dict['filename'] = temp
        else:
            temp = self.__cmd_files_path+'\\'+temp_filename
            one_data_dict['filename'] = temp

        return one_data_dict

    def set_cmd_path(self):
        for (path, dir, files) in os.walk(self.__data_path):
            temp = path.split('\\')[-1]
            if temp == 'cmd':
                return path
        try:
            raise Exception("cmd 폴더 명이 존재하지 않습니다.")
        except Exception as e:
            print(e)
            return None

    def set_nonecmd_path(self):
        for (path, dir, files) in os.walk(self.__data_path):
            temp = path.split('\\')[-1]
            if temp == 'noncmd':
                return path
        try:
            raise Exception("noncmd 폴더 명이 존재하지 않습니다.")
        except Exception as e:
            print(e)
            return None


class GenLabels():
    def __init__(self):
        self.__index_val = int(0)
        self.__curr_label = int(0)
        self.label_block = LABEL_BLOCK_SIZE
        # 얘 때문에 만든 클래스
        self.__label_dict = self.gen_label_dict()

    def return_label_value(self, idx_num):
        self.__index_val = idx_num
        temp = self.__index_val//self.label_block
        target_name = list(self.__label_dict.keys())[temp]
        self.__curr_label = self.__label_dict[target_name]

    def get_label(self, idx_num):
        self.return_label_value(idx_num)
        return self.__curr_label
    
    def get__index(self):
        return self.__index_val

    def gen_label_dict(self):
        return_dict = dict()
        for one in LabelsKorEng:
            return_dict[one.name] = one.value
        return_dict['NONE'] = 0
        return return_dict






def json_read_main():
    result = CW_load_json_files()
    result_list = refine_loaded_data(result)
    for r in result_list:
        print(r)

    print(len(result_list))

    return





if __name__ == '__main__':
    print('hello, world~!!')
    json_read_main()

