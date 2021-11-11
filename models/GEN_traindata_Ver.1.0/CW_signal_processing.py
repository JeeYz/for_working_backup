
from global_variables import *
import signal_processing as spro
import global_variables as gv

def receive_files_list(input_data_list):
    num = 0

    for one_file_dict in input_data_list :
        gen_signal_data(one_file_dict)
        num += 1
        print(num, end='\r')


def gen_signal_data(one_file_dict):
    one_filename = one_file_dict['filename']

    print_sent = str()

    try:
        with wave.open(one_filename, 'rb') as wf:
            parameters = wf.getparams()
    except FileNotFoundError as e:
        print(e)
        return

    a = wavio.read(one_filename)
    print_sent = str(a)+' '

    data0 = a.data[:, 0]
    data1 = a.data[:, 1]

    try:
        if data0.all() == data1.all():
            print_sent += "True "
        else:
            print_sent += "False"
            raise Exception("양 채널의 음성 데이터가 같지 않습니다.")
    except Exception as e:
        print(e)

    curr_data = data0 

    if len(curr_data) < gv.RESOURCE_FULL_SIZE:
        temp_gap = gv.RESOURCE_FULL_SIZE-a.data.shape[0]
        zero_pad = np.zeros(temp_gap, dtype=a.data.dtype)
        curr_data = np.append(zero_pad, curr_data)

    # standardization
    curr_data = (curr_data-np.mean(curr_data))/np.std(curr_data)
    curr_data = np.array(curr_data, dtype=gv.TRAIN_DATA_TYPE)

    new_data_len = int(len(curr_data)/a.rate*gv.TARGET_SAMPLE_RATE)

    # curr_data = sps.resample(curr_data, new_data_len)
    curr_data = sps.decimate(curr_data, 3)

    if len(curr_data) != gv.RESOURCE_SECS*gv.TARGET_SAMPLE_RATE:
        print(len(curr_data))
        raise Exception("길이가 다릅니다.")
    
    temp_dict = gv.GLOBAL_CW_TRAINDATA.gen_file_data_dict(
        label=one_file_dict['file_label'],
        length=len(curr_data),
        data=curr_data
    )

    one_file_dict['file_data'].append(temp_dict)

    print_sent = print_sent+str(np.max(curr_data))+' '+str(np.min(curr_data))
    print(f'{print_sent}', end='\r')


def gen_sig_data_2(self, input_dict):
    one_filename = input_dict['filename']

    print_sent = str()

    try:
        with wave.open(one_filename, 'rb') as wf:
            parameters = wf.getparams()
    except FileNotFoundError as e:
        print(e)
        return

    a = wavio.read(one_filename)
    print_sent = str(a)+' '

    data0 = a.data[:, 0]
    data1 = a.data[:, 1]

    try:
        if data0.all() == data1.all():
            print_sent += "True "
        else:
            print_sent += "False"
            raise Exception("양 채널의 음성 데이터가 같지 않습니다.")
    except Exception as e:
        print(e)

    curr_data = data0 

    if len(curr_data) < gv.RESOURCE_FULL_SIZE:
        temp_gap = gv.RESOURCE_FULL_SIZE-a.data.shape[0]
        zero_pad = np.zeros(temp_gap, dtype=a.data.dtype)
        curr_data = np.append(zero_pad, curr_data)

    # standardization
    curr_data = (curr_data-np.mean(curr_data))/np.std(curr_data)
    curr_data = np.array(curr_data, dtype=gv.TRAIN_DATA_TYPE)

    new_data_len = int(len(curr_data)/a.rate*gv.TARGET_SAMPLE_RATE)

    # curr_data = sps.resample(curr_data, new_data_len)
    curr_data = sps.decimate(curr_data, 3)

    if len(curr_data) != gv.RESOURCE_SECS*gv.TARGET_SAMPLE_RATE:
        print(len(curr_data))
        raise Exception("길이가 다릅니다.")
    
    return_dict = dict()    
    return_dict['filename'] = input_dict['filename']
    return_dict['file_label'] = input_dict['label']
    return_dict['data'] = list()
    return_dict['data'].append(curr_data)

    print_sent = print_sent+str(np.max(curr_data))+' '+str(np.min(curr_data))
    print(f'{print_sent}', end='\r')

    return return_dict


    




if __name__ == '__main__':
    print('hello, world~!!')
