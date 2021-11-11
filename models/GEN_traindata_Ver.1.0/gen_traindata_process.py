
from global_variables import *


class TrainData():
    def __init__(self, **kwargs):
        if "json_file" in kwargs.keys():
            self.json_file = kwargs['json_file']
        else:
            self.json_file = None
        
        if "data_type" in kwargs.keys():
            self.data_type = kwargs['data_type']
        
        # main variable
        self.__train_data_info_dict = None
        self.__json_data = None
        self.__target_data_list = None

        # initialization
        self.__load_json_file()
        self.__init_targer_files_list()


    def set_train_data(self, input_data):
        self.__train_data_info_dict = input_data

    def get_train_data(self):
        return self.__train_data_info_dict

    def get_json_data(self):
        return self.__json_data

    def __load_json_file(self):
        loaded_data = json.load(self.json_file)
        self.__json_data = loaded_data

    def __init_targer_files_list(self):
        temp_dict = dict()
        temp_dict['speaker'] = self.__json_data['speaker']
        temp_dict['data_type'] = self.data_type
        temp_dict['files'] = list()

        for one_file in self.__json_data['files']:
            one_file['filename'] = self.__json_data['path']+one_file['filename']
            result = cwsig.gen_sig_data_2(one_file)
            temp_dict['files'].append(result)


def gen_one_npz_file_process():
    return


def gen_traindata_process():

    return







