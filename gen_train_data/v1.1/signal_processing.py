from global_variables import *
import global_variables as gv


def gen_sig_data_2(input_dict):
    
    one_filename = input_dict['filename']

    try:
        with wave.open(one_filename, 'rb') as wf:
            parameters = wf.getparams()
    except FileNotFoundError as e:
        print(e)
        return

    a = wavio.read(one_filename)

    data0 = a.data[:, 0]
    data1 = a.data[:, 1]

    return_dict_0 = gen_data_dictionary(data0, input_dict, a)
    return_dict_1 = gen_data_dictionary(data1, input_dict, a)

    return_list = list()
    return_list = [return_dict_0, return_dict_1]

    return return_list



def gen_data_dictionary(curr_data, input_dict, stereo_data):

    one_filename = input_dict['filename']

    print_sent = str()
    print_sent = str(stereo_data)+' '

    if len(curr_data) < gv.RESOURCE_FULL_SIZE:
        temp_gap = gv.RESOURCE_FULL_SIZE-stereo_data.data.shape[0]
        zero_pad = np.zeros(temp_gap, dtype=stereo_data.data.dtype)
        curr_data = np.append(zero_pad, curr_data)

    # standardization
    # curr_data = (curr_data-np.mean(curr_data))/np.std(curr_data)
    # curr_data = np.array(curr_data, dtype=gv.TRAIN_DATA_TYPE)

    new_data_len = int(len(curr_data)/stereo_data.rate*gv.TARGET_SAMPLE_RATE)

    # curr_data = sps.resample(curr_data, new_data_len)
    curr_data = sps.decimate(curr_data, 3)

    if len(curr_data) != gv.RESOURCE_SECS*gv.TARGET_SAMPLE_RATE:
        print(len(curr_data))
        raise Exception("길이가 다릅니다.")
    
    return_dict = dict()    
    return_dict['filename'] = one_filename
    return_dict['file_label'] = input_dict['label']
    return_dict['file_data'] = curr_data

    print_sent = print_sent+str(np.max(curr_data))+' '+str(np.min(curr_data))
    print(f'{print_sent}', end='\r')

    return return_dict








