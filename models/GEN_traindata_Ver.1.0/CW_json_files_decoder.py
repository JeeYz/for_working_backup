
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
                    gen_label_data(
                        data_dict_list, 
                        one_result['dataID']
                    )


def find_files_path(input_filename):
    return_filename = None
    for (path, dir, files) in os.walk(CWdata_path):
        for filename in files:
            if input_filename==filename: 
                temp = path+"\\"+filename
                return_filename = temp
                break
    
    try:
        if os.path.isfile(temp):
            pass
        else:
            raise Exception("현재 찾은 파일이 존재하지 않습니다.")
    except Exception as e:
        print(e)
        
    return return_filename 


def gen_label_data(data_dict_list, speaker_name):
    result_list = list()

    for one_list in data_dict_list:
        data_dict = one_list['value']
        gl_class = GenLabels()

        if len(data_dict) is 1:
            non_cmd_flag = True 
        else:
            non_cmd_flag = False

        for i, one_filename in enumerate(data_dict):
            # temp_dict = GLOBAL_CW_TRAINDATA.set_whole_data_dict()
            # temp = one_filename['file_name']
            # temp_dict['speaker'] = speaker_name
            # temp_dict['filename'] = temp

            # if non_cmd_flag is True:
            #     temp_dict['file_label'] = 0
            # else:
            #     temp_dict['file_label'] = gl_class.get_label(i)

            # return_dict = find_files_path(temp_dict) 
            # GLOBAL_CW_TRAINDATA.set_whole_data_list(return_dict)
            # result_list.append(return_dict)

            # 위 아래 비교

            temp_filename = one_filename['file_name']

            if non_cmd_flag is True:
                temp_file_label = 0
            else:
                temp_file_label = gl_class.get_label(i)

            return_filename = find_files_path(temp_filename) 

            try:
                return_dict = GLOBAL_CW_TRAINDATA.set_whole_data_dict(
                    speaker=speaker_name,
                    filename=return_filename,
                    file_label=temp_file_label,
                )
            except TypeError as te:
                print(te)
                print("다음 for문으로 넘어 갑니다.")
                continue

            GLOBAL_CW_TRAINDATA.set_whole_data_list(return_dict)
        
        print(i+1, end=' ')


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
    refine_loaded_data(result)





if __name__ == '__main__':
    print('hello, world~!!')
    json_read_main()

