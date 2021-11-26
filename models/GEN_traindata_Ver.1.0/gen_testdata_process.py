

from global_variables import *
from CW_class_data import TrainData
import global_variables as gv

padding_size = 10000
hy_target_filename = 'test_Ver.2.4.npz'

def add_zero_padding(input_data):
    zero_padding = np.zeros(padding_size, dtype=gv.TRAIN_DATA_TYPE)
    result_data = np.append(zero_padding, input_data)
    result = np.append(result_data, zero_padding)

    return result


def gen_sig_data(input_list):

    for one_dict in input_list:
        filename = one_dict['filename']
        sr, data = wavfile.read(filename)
        one_dict['data'] = add_zero_padding(data)


def gen_files_dict(files_list):
    return_list = list()
    label_list = list(test_pnc_dict.keys())
    
    for one_file in files_list:
        temp_dict = dict()
        parsing_temp = one_file.split('\\')
        temp_label = parsing_temp[-2]

        if temp_label in label_list:
            temp_dict['label'] = test_pnc_dict[temp_label]
            temp_dict['filename'] = one_file
        else:
            temp_dict['label'] = 0
            temp_dict['filename'] = one_file

        return_list.append(temp_dict)
    
    return return_list

    
def detect_index(input_list):
    for one_file in input_list:
        data = one_file['data']
        result = trigal.signal_trigger_algorithm_for_decode(data) 
        one_file['data'] = result


def gen_numpy_file(input_list):
    data_list = list()
    label_list = list()

    for one_file in input_list:
        data_list.append(one_file['data'])
        label_list.append(one_file['label'])

    for one_data in data_list:
        if len(one_data) != gv.FULL_SIZE:
            raise("데이터 길이가 정해진 길이가 아닙니다.")
    
    data_list = np.asarray(data_list, dtype=TRAIN_DATA_TYPE)
    label_list = np.asarray(label_list, dtype=np.int8)

    target_file_name = CWdata_path+'\\'+hy_target_filename

    np.savez(
        target_file_name, 
        data=data_list,
        label=label_list,
    )
    return


def gen_testdata():
    test_files = fpro.find_data_files(
        'C:\\temp\\test\\',
        '.wav',
    )

    result_list = gen_files_dict(test_files)
    gen_sig_data(result_list)

    detect_index(result_list)
    gen_numpy_file(result_list)

    return





if __name__ == '__main__':
    print('hello, world~!!')
    
    gen_testdata()
    
    
