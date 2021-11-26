
from pickle import NONE
from global_variables import *
import global_variables as gv
# from global_variables import GLOBAL_CW_TRAINDATA


def CW_load_json_files():
    files_list = fpro.find_data_files(gv.CWdata_path, '.json')
    new_files_list = list()

    for one_file in files_list:
        if '$$' not in one_file:
            new_files_list.append(one_file)

    whole_json_data = list()

    for one_jsonfile in new_files_list:
        print(one_jsonfile)
        with open(one_jsonfile, 'r', encoding='utf-8') as jf:
            loaded_jsondata = json.load(jf)
        whole_json_data.append(loaded_jsondata)

    return whole_json_data


def refine_loaded_data(whole_json_data):
    temp_list = list()

    for one_jsonfile in whole_json_data:
        temp_result = one_jsonfile['result']
        # print(one_jsonfile)
        
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
    for (path, dir, files) in os.walk(gv.CWdata_path):
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
            # temp_dict = gv.GLOBAL_CW_TRAINDATA.set_whole_data_dict()
            # temp = one_filename['file_name']
            # temp_dict['speaker'] = speaker_name
            # temp_dict['filename'] = temp

            # if non_cmd_flag is True:
            #     temp_dict['file_label'] = 0
            # else:
            #     temp_dict['file_label'] = gl_class.get_label(i)

            # return_dict = find_files_path(temp_dict) 
            # gv.GLOBAL_CW_TRAINDATA.set_whole_data_list(return_dict)
            # result_list.append(return_dict)

            # 위 아래 비교

            temp_filename = one_filename['file_name']

            if non_cmd_flag is True:
                temp_file_label = 0
            else:
                temp_file_label = gl_class.get_label(i)

            return_filename = find_files_path(temp_filename) 

            try:
                return_dict = gv.GLOBAL_CW_TRAINDATA.gen_whole_data_dict(
                    speaker=speaker_name,
                    filename=return_filename,
                    file_label=temp_file_label,
                )
            except TypeError as te:
                print(te)
                print("다음 for문으로 넘어 갑니다.")
                continue

            gv.GLOBAL_CW_TRAINDATA.set_whole_data_list(return_dict)
        


class GenLabels():
    def __init__(self):
        self.__index_val = int(0)
        self.__curr_label = int(0)
        self.label_block = gv.LABEL_BLOCK_SIZE
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
        for one in gv.LabelsKorEng:
            return_dict[one.name] = one.value
        return_dict['NONE'] = 0
        return return_dict


class WriteJsonFiles():
    def __init__(self):
        self.label_dict = GenLabels()
        self.__target_json_data = list()
        self.wav_files_list = fpro.find_data_files(CWdata_path, '.wav')
        self.json_files_data = CW_load_json_files()
        print(len(self.json_files_data))
        self.__refine_source_json_data()
        self.root_dir = CWdata_path

    def get_target_json(self):
        return self.__target_json_data

    def set_target_json(self, input_data):
        self.__target_json_data = input_data

    def print_target_json(self):
        print(json.dumps(
                self.__target_json_data,
                sort_keys=False, 
                indent=4, 
                default=str, 
                ensure_ascii=False
        ))

    def print_target_json_length(self):
        print(len(self.__target_json_data))

    def __find_target(self, target_filename):
        for one_file in self.wav_files_list:
            if target_filename in one_file:
                parsing_temp = one_file.split('\\')
                target_path = '\\'.join(parsing_temp[:-1])+'\\'
                target_type = parsing_temp[-3]
                json_filename_temp = '$$'+parsing_temp[-2]+'_'+target_type+'.json'
                return target_path, target_type, json_filename_temp
        
        try:
            raise Exception('아무것도 찾아내지 못했습니다.')
        except Exception:
            print('None을 반환합니다.')
            return None, None, None
        

    def __return_target_keys(self, one_result):
        list_keys_in_result = list(one_result.keys())

        target_keys_temp = list()

        for one_key in list_keys_in_result:
            if 'name_' in one_key:
                # target_key_temp = one_key
                target_keys_temp.append(one_key)

        one_key = target_keys_temp[0]
        target_dict_temp = one_result[one_key]
        target_data_dict_temp = target_dict_temp['data'][0]
        target_value_temp = target_data_dict_temp['value']

        target_filename = target_value_temp[0]['file_name']

        target_path, target_type, json_filename_temp = self.__find_target(target_filename)
        
        temp_dict = dict()

        temp_dict['speaker'] = one_result['dataID']
        temp_dict['path'] = target_path
        temp_dict['type'] = target_type
        temp_dict['json_filename'] = json_filename_temp
        temp_dict['files'] = list()

        for one_key in target_keys_temp:
            target_dict_temp = one_result[one_key]
            target_data_dict_temp = target_dict_temp['data'][0]
            target_value_temp = target_data_dict_temp['value']

            for i, one_value in enumerate(target_value_temp):
                temp_one_file_dict = dict()
                return_label = self.label_dict.get_label(i)

                if 'noncmd_' in target_type:
                    return_label = 0
                
                temp_one_file_dict['filename'] = one_value['file_name']
                temp_one_file_dict['label'] = return_label

                
                temp_dict['files'].append(temp_one_file_dict)
            
        return temp_dict
        
    def __refine_source_json_data(self):
        for one_json_data in self.json_files_data:
            for one_result in one_json_data['result']:
                return_dict = self.__return_target_keys(one_result)
                self.__target_json_data.append(return_dict)

    def write_each_json_files(self):
        whole_info_json = list()
        speakers_list = list()
        for one_json in self.__target_json_data:
            target_file = one_json['path']+'\\'+one_json['json_filename']
            with open(target_file, 'w', encoding='utf-8') as fw:
                json.dump(
                    one_json, 
                    fw, 
                    indent='\t', 
                    ensure_ascii=False
                )
            print('generate json file... {filename}'.format(filename=target_file))

            speakers_list.append(int(one_json['speaker']))

            curr_speaker = None
            for one_info in whole_info_json:
                if one_info['speaker'] == one_json['speaker'] :
                    curr_speaker = one_info['speaker']
                    new_filename = one_json['path']+one_json['json_filename']
                    temp_dict['files'].append(new_filename)
                   
            if curr_speaker is None:
                temp_dict = dict()
                temp_dict['speaker'] = one_json['speaker']
                temp_dict['files'] = list()
                new_filename = one_json['path']+one_json['json_filename']
                temp_dict['files'].append(new_filename)
                whole_info_json.append(temp_dict)

        whole_info_path = CWdata_path+'\\'+'$$whole_data_info.json'     
        with open(whole_info_path, 'w', encoding='utf-8') as fw:
            json.dump(
                whole_info_json,
                fw,
                indent='\t',
                ensure_ascii=False,
            )
        print('generate json file... {filename}'.format(filename=whole_info_path))
        





def json_read_main():
    cwjson = WriteJsonFiles()
    cwjson.print_target_json_length()
    cwjson.write_each_json_files()

    return





if __name__ == '__main__':
    print('hello, world~!!')
    json_read_main()


