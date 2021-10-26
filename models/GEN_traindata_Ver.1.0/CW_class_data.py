
from wave import Error
from global_variables import *


class TrainData():
    def __init__(self, **kwargs):

        if "kind" in kwargs.keys():
            self.data_kind = kwargs['kind']
        else:
            self.data_kind = 'train'

        # 얘가 중심 변수
        self.__traindata_dict = self.initialization()


    def initialization(self):
        temp_dict = dict()
        temp_dict['kind_of_data'] = self.data_kind
        temp_dict['whole_data'] = list()
        return temp_dict


    def set_whole_data_dict(self, *args, **kwargs):
        temp_dict = dict()

        if 'speaker' in kwargs.keys():
            temp_dict['speaker'] = kwargs['speaker']
        else:
            temp_dict['speaker'] = None

        if 'file_label' in kwargs.keys():
            temp_dict['file_label'] = kwargs['file_label']
        else:
            temp_dict['file_label'] = None

        if 'filename' in kwargs.keys():
            temp_dict['filename'] = kwargs['filename']
        else:
            temp_dict['filename'] = None

        temp_dict['file_data'] = list()

        return temp_dict


    def set_file_data_dict(self, *args, **kwargs):
        temp_dict = dict()
        if 'label' in kwargs.keys():
            temp_dict['data_label'] = kwargs['label']
        else:
            temp_dict['data_label'] = None

        if 'length' in kwargs.keys():
            temp_dict['data_length'] = kwargs['length']
        else:
            temp_dict['data_length'] = None

        if 'start_index' in kwargs.keys():
            temp_dict['start_index'] = kwargs['start_index']
        else:
            temp_dict['start_index'] = None

        if 'end_index' in kwargs.keys():
            temp_dict['end_index'] = kwargs['end_index']
        else:
            temp_dict['end_index'] = None

        if 'gap_start_end' in kwargs.keys():
            temp_dict['gap_start_end'] = kwargs['gap_start_end']
        else:
            temp_dict['gap_start_end'] = None

        if 'auged_position' in kwargs.keys():
            temp_dict['auged_position'] = kwargs['auged_position']
        else:
            temp_dict['auged_position'] = None

        if 'auged_boolean' in kwargs.keys():
            temp_dict['auged_boolean'] = kwargs['auged_boolean']
        else:
            temp_dict['auged_boolean'] = None

        if 'data' in kwargs.keys():
            temp_dict['auged_data'] = kwargs['data']
        else:
            temp_dict['auged_data'] = None


        return temp_dict


    def get_traindata(self):
        return self.__traindata_dict


    def set_traindata_class(self, input_dict):
        self.__traindata_dict = input_dict


    def set_whole_data_list(self, input_dict):
        try:
            self.__traindata_dict['whole_data'].append(input_dict)
        except Error as e:
            print(e)
            self.__traindata_dict['whole_data'] = list()
            self.__traindata_dict['whole_data'].append(input_dict)

    def set_file_data_list(self, input_dict):
        try:
            temp_target = self.__traindata_dict['whole_data']
            temp_target.append(input_dict)
        except Error as e:
            print(e)
            raise Exception("훈련데이터 구조가 정상적이지 않습니다.")
        

    def print_whole_data_length(self):
        print("\n")
        temp=self.__traindata_dict['whole_data']
        print("CW train data volume : {num}".format(num=len(temp)))
        print("\n")


    def print_whole_train_data_info(self):
        print(json.dumps(
                        self.__traindata_dict,
                        sort_keys=False, 
                        indent=4, 
                        default=str, 
                        ensure_ascii=False
        ))



