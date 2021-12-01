

from global_variables import *
from CW_class_data import TrainData
import global_variables as gv

padding_size = 10000
hy_target_filename = 'test_Ver.2.4.npz'

target_cw_test_path = 'C:\\temp\\CW_test\\'


def standardization_data(input_data):
    return (input_data-np.mean(input_data))/np.std(input_data)


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

        print(temp_dict['label'])

        return_list.append(temp_dict)
    
    return return_list

    
def detect_index(input_list):
    for one_file in input_list:
        data = one_file['data']
        data = standardization_data(data)
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

    
    
def load_json_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        loaded = json.load(f)

    return loaded


def return_files_list(json_data):
    return_list = list()
    return_label = list()
    temp_path = json_data['path']

    for one_file in json_data['files']:
        temp = temp_path+one_file['filename']
        return_list.append(temp)
        return_label.append(int(one_file['label']))
    
    return return_list, return_label

    
def gen_data_and_convert(input_list):
    return_list = list()
    for one_file in input_list:
        temp_dict = dict()
        temp_dict['filename'] = one_file
        temp_dict['label'] = 'none'
        return_dict = cwsig.gen_sig_data_2(temp_dict)
        ret_data = trigal.signal_trigger_algorithm_for_decode(return_dict['file_data'])
        if len(ret_data) != gv.FULL_SIZE:
            print('\n')
            print(len(ret_data))
            print(one_file)
            print('\n')
        return_list.append(ret_data)

    return return_list
    
    
def write_numpy_data(data_list, labels_list, json_data):
    target_npz_file = str(json_data['speaker'])+'_'+json_data['type']+'_'
    target_npz_file = target_npz_file+'for_test.npz'
    target_npz_file = target_cw_test_path+target_npz_file

    np.savez(
        target_npz_file,
        data=data_list,
        label=labels_list,
    )
    
    return target_npz_file


def gen_testdata_for_cw(input_list):

    speakers_list = list()

    for one in input_list:
        temp = one.split('\\')
        filename = temp[-1]
        speaker = filename.split('_')[1]
        speakers_list.append(speaker)

    files_list = fpro.find_data_files(CWdata_path, '.json')

    target_jsons = list()

    for one_speaker in speakers_list:
        for one_json in files_list:
            if one_speaker in one_json:
                target_jsons.append(one_json)

    result_list = list()

    for one_json in target_jsons:
        json_data = load_json_data(one_json)
        return_list, return_labels = return_files_list(json_data)
        data_list = gen_data_and_convert(return_list)
        return_file = write_numpy_data(data_list, return_labels, json_data)
        # print(return_file)
        return_list.append(return_file)

    return result_list



if __name__ == '__main__':
    print('hello, world~!!')
    
    gen_testdata()
    
    
