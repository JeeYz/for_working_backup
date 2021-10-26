
from global_variables import *
import signal_processing as spro

def receive_files_list(input_files_list):
    num = 0

    for one_file_dict in input_files_list:
        gen_signal_data(one_file_dict)
        print(num, end=' ')
        num += 1

    return input_files_list


def gen_signal_data(one_file_dict):
    one_filename = one_file_dict['filename']

    try:
        with wave.open(one_filename, 'rb') as wf:
            parameters = wf.getparams()
    except FileNotFoundError as e:
        print(e)
        return

    a = wavio.read(one_filename)
    print(a, end=' ')

    data0 = a.data[:, 0]
    data1 = a.data[:, 1]

    try:
        if data0.all() == data1.all():
            print("True", end=' ')
        else:
            print("False", end=' ')
            raise Exception("양 채널의 음성 데이터가 같지 않습니다.")
    except Exception as e:
        print(e)

    curr_data = data0 

    if len(curr_data) < RESOURCE_FULL_SIZE:
        temp_gap = RESOURCE_FULL_SIZE-a.data.shape[0]
        zero_pad = np.zeros(temp_gap, dtype=a.data.dtype)
        curr_data = np.append(zero_pad, curr_data)

    curr_data = (curr_data-np.mean(curr_data))/np.std(curr_data)
    curr_data = np.array(curr_data, dtype=TRAIN_DATA_TYPE)

    new_data_len = int(len(curr_data)/a.rate*TARGET_SAMPLE_RATE)

    # curr_data = sps.resample(curr_data, new_data_len)
    curr_data = sps.decimate(curr_data, 3)

    if len(curr_data) != RESOURCE_SECS*TARGET_SAMPLE_RATE:
        print(len(curr_data))
        raise Exception("길이가 다릅니다.")
    
    one_file_dict['data'] = list()
    one_file_dict['data'].append(curr_data)
    one_file_dict['train'] = True

    print(np.max(curr_data), np.min(curr_data))

    return one_file_dict






if __name__ == '__main__':
    print('hello, world~!!')
