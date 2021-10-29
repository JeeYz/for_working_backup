
from wave import Error
from global_variables import *


class TrainData():
    def __init__(self, *args, **kwargs):

        if 'numpy_path' in kwargs.keys():
            self.numpy_filepath = kwargs['numpy_path']
        else:
            self.numpy_filepath = None

        if 'json_path' in kwargs.keys():
            self.json_filepath = kwargs['json_path']
        else:
            self.json_filepath  = None

        if 'dtype' in kwargs.keys():
            self.global_dtype = kwargs['dtype']
        else:
            self.global_dtype = None

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
            temp_dict['auged_boolean'] = False

        if 'data' in kwargs.keys():
            temp_dict['data'] = kwargs['data']
        else:
            temp_dict['data'] = None


        return temp_dict


    def get_traindata(self):
        return self.__traindata_dict


    def get_whole_data_list(self):
        return self.__traindata_dict['whole_data']


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


    def generate_json_file(self):
        with open(self.json_filepath, 'w', encoding='utf-8') as jwf:
            json.dump(self.__traindata_dict, jwf, indent='\t')


    def generate_numpy_file(self):
        data_list = list()
        label_list = list()

        whole_data_list = self.__traindata_dict['whole_data']

        for one_file in whole_data_list:
            for one_data in one_file['file_data']:
                data_list.append(one_data['data'])
                label_list.append(one_data['data_label'])

        data_list = np.asarray(data_list, dtype=self.global_dtype)
        label_list = np.asarray(label_list, dtype=np.int8)

        np.savez(
            self.numpy_filepath, 
            data=data_list,
            label=label_list,
        )
        

class DecodingData():
    global PREPRO_SHIFT_SIZE
    global PREPRO_FRAME_SIZE
    def __init__(self):
        # private
        self.__target_data = list()
        # public
        self.condition_num = 0
        # simple global variable
        self.stack_data = list()

    def get_target_data(self):
        return self.__target_data

    def set_none_target_data(self):
        self.__target_data = list()

    def set_none_stack_data(self):
        self.stack_data = list()

    def set_condition_num_zero(self):
        self.condition_num = 0

    def standardization_data(self):
        data = self.__target_data
        self.__target_data = (data-np.mean(data))/np.std(data)

    def add_a_sec_condition(self):
        self.condition_num+=1

    def set_target_data(self, input_data):
        self.__target_data = input_data





if __name__ == '__main__':
    print(FULL_SIZE)
    print(json_file_CWdata)
    print(numpy_traindata_file_CWdata)
