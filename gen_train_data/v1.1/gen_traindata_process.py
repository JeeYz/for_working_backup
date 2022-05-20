
import global_variables as gv
import augment_processing as augp


class OneTrainData():
    def __init__(self, **kwargs):
        if "json_file" in kwargs.keys():
            self.json_file = kwargs['json_file']
        else:
            self.json_file = None
        
        if "data_type" in kwargs.keys():
            self.data_type = kwargs['data_type']
        else:
            self.data_type = None
        
        # main variable
        self.__train_data_info_dict = None
        self.__json_data = None

        # initialization
        self.__load_json_file()
        self.file_type = self.__json_data['type']

    def set_train_data(self, input_data):
        self.__train_data_info_dict = input_data

    def get_train_data(self):
        return self.__train_data_info_dict

    def get_json_data(self):
        return self.__json_data

    def __load_json_file(self):
        with open(self.json_file, 'r', encoding='utf-8') as fr:
            loaded_data = json.load(fr)
        self.__json_data = loaded_data

    def print_train_data_info(self):
        print(json.dumps(
            self.__train_data_info_dict,
            sort_keys=False, 
            indent=4, 
            default=str, 
            ensure_ascii=False
        ))

    def __create_target_dir(self):
        try:
            if not os.path.exists(npz_target_path):
                os.makedirs(npz_target_path)
        except OSError:
            print ('Error: Creating directory. ' +  npz_target_path)

    def __return_npz_name(self):
        temp_speaker = self.__train_data_info_dict[0]['speaker']
        return_name = 'speaker_'+str(temp_speaker)+'_'+self.file_type+'_traindata.npz'
        return return_name

    def write_numpy_file(self):
        self.__create_target_dir()

        data_list = list()
        label_list = list()

        whole_data_list = self.__train_data_info_dict

        for one_file in whole_data_list:
            for one_data in one_file['file_data']:
                data_list.append(one_data['data'])
                label_list.append(one_data['data_label'])

        num = 0
        for one in data_list:
            if len(one) != FULL_SIZE:
                print(one, len(one))
                num+=1

        data_list = np.asarray(data_list, dtype=TRAIN_DATA_TYPE)
        label_list = np.asarray(label_list, dtype=np.int8)

        target_file_name = npz_target_path+self.__return_npz_name()

        np.savez(
            target_file_name, 
            data=data_list,
            label=label_list,
        )


def gen_whole_data_dict(*args, **kwargs):
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


def gen_file_data_dict(*args, **kwargs):
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

    
def break_point(input_val):
    if input_val == 'y':
        pass
    elif input_val == 'n':
        print("프로그램을 종료합니다...")
        exit()
    else:
        break_point(input("reinput answer?? :"))


def gen_one_npz_file_process(target_json_file):
    cw_train = OneTrainData(
        json_file = target_json_file,
        data_type = 'train',
    )

    json_data = cw_train.get_json_data()
    
    target_path = json_data['path']

    one_npz_data_list = list()

    for one_file_dict in json_data['files']:
        one_file_dict['filename'] = target_path+one_file_dict['filename']
        # print(one_file_dict['filename'])
        return_dict = cwsig.gen_sig_data_2(one_file_dict)

        w_dict = gen_whole_data_dict(
            speaker=json_data['speaker'],
            file_label=one_file_dict['label'],
            filename=one_file_dict['filename'],
        )
        f_dict = gen_file_data_dict(
            label=one_file_dict['label'],
            data=return_dict['file_data'],
        )
        w_dict['file_data'].append(f_dict)
        one_npz_data_list.append(w_dict)

    cw_train.set_train_data(one_npz_data_list)
    # cw_train.print_train_data_info()
    # break_point(input("answer?? :"))

    trigal.apply_trigger_algorithm(
        one_npz_data_list,
        cw_train.data_type,
    )
    
    cw_train.set_train_data(one_npz_data_list)
    # cw_train.print_train_data_info()
    # break_point(input("answer?? :"))

    augp.aug_only_time_stretch(one_npz_data_list)

    cw_train.set_train_data(one_npz_data_list)
    # cw_train.print_train_data_info()
    # break_point(input("answer?? :"))

    cw_train.write_numpy_file()
    # break_point(input("answer?? :"))

    return

    
def gen_json_for_npz():
    npz_files_list = fpro.find_data_files(
        npz_target_path,
        '.npz',
    )

    target_file = CWdata_path+"$$npz_data.json"

    with open(target_file, 'w', encoding='utf-8') as fw:
        json.dump(
            npz_files_list, 
            fw, 
            indent='\t', 
            ensure_ascii=False
        )

    return


def gen_traindata_process(**kwargs):
    #
    if "filepath" in kwargs.keys():
        filepath = kwargs['filepath']
    else:
        filepath = None
        print("Nothing has come...")
    
    if "json_file" in kwargs.keys():
        json_file = kwargs['json_file']
    else:
        json_file = None

    json_filepath = filepath+'/'+json_file
    print(json_filepath)

    # 
    with open(json_filepath, 'r', encoding='utf-8') as fr:
        loaded_whole_json = json.load(fr)

    for one_speaker in loaded_whole_json:
        for one_file in one_speaker['files']:
            gen_one_npz_file_process(one_file)

    gen_json_for_npz()

    return






if __name__ == '__main__':
    print('hello, world~!!')

    gen_traindata_process(
        filepath=CWdata_path,
        json_file=whole_data_json_filename,
    )
    
    
    
