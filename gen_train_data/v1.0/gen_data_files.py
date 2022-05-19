
from global_variables import *


def write_numpy(data_list):
    data_list = refine_data(data_list)

    for one_data in data_list:
        one_data['data'] = np.asarray(one_data['data'])
        one_data['label'] = np.asarray(one_data['label'])

        print("*** number of generating data ***")
        print("number of total data : {num}".format(num=len(one_data['data'])))
        print("number of total data : {num}".format(num=len(one_data['label'])))
        print('\n')

        cond1 = len(one_data['data'])
        cond2 = len(one_data['label'])

        if cond1 != cond2:
            raise Exception("데이터와 레이블의 개수가 일치하지 않습니다.")

        if one_data['kind'] is 'train':
            np.savez(
                # numpy_traindata_files_path,
                # numpy_traindata_files_path_zero,
                numpy_traindata_file_CWdata, 
                data=one_data['data'],
                label=one_data['label'],
            )
        elif one_data['kind'] is 'test':
            np.savez(
                numpy_testdata_files_path,
                data=one_data['data'],
                label=one_data['label'],
            )
            
    return


def refine_data(data_list):
    train_dict = dict()
    test_dict = dict()

    train_dict['data'] = list()
    train_dict['label'] = list()
    train_dict['kind'] = 'train'

    test_dict['data'] = list()
    test_dict['label'] = list()
    test_dict['kind'] = 'test'

    for one_file in data_list:
        # for one_file in one_list:
        for one_data in one_file['data']:
            temp_label = one_data['label']
            if one_file['train'] is True:
                train_dict['data'].append(one_data['data'])
                train_dict['label'].append(temp_label)
            else:
                test_dict['data'].append(one_data['data'])
                test_dict['label'].append(temp_label)

    return [train_dict, test_dict]






