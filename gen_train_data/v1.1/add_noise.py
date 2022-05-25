import global_variables as gv
from files_module import find_data_files as fdf
from generate_zeroth_data import regenerate_json_file

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.io import wavfile



noise_files_list = list()
noise_data = list()



def gen_noised_data_main():
    pre_generate_expo_noise()

    all_npz_result_list = fdf(gv.npz_target_path, ".npz")

    for one_npz_file in all_npz_result_list:
        return_data, return_label = add_noise_expo(one_npz_file)
        write_npz_file(return_data, return_label, one_npz_file)

    for one_npz_file in all_npz_result_list:
        return_data, return_label = add_noise_gaussian(one_npz_file)
        write_npz_file_gaussian(return_data, return_label, one_npz_file)

    regenerate_json_file()

    return





def pre_generate_expo_noise():
    global noise_files_list
    global noise_data 

    noise_files_list = fdf(gv.noise_data_path, ".wav")

    temp_result = list()
    for one_file in noise_files_list:
        if "음성" not in one_file:
            temp_result.append(one_file)

    return_data = list()
    for one_file in temp_result:
        sr, data = wavfile.read(one_file)
        return_noise_with_block = gen_data_by_block(data)
        return_data.extend(return_noise_with_block)

    noise_data = return_data

    return 




def gen_data_by_block(input_data):
    return_list = list()

    num_of_block = len(input_data)//gv.FULL_SIZE

    for i in range(num_of_block):
        temp_data = input_data[i*gv.FULL_SIZE:(i+1)*gv.FULL_SIZE]

        temp_data = np.array(temp_data, dtype=np.float32)
        temp_data = normalization_data(temp_data)

        # draw_single_graph(temp_data)
        return_list.append(temp_data)

    return return_list




def normalization_data(input_data):
    result = (input_data - np.min(input_data))/(np.max(input_data) - np.min(input_data))

    result = result*4-2

    return result




def add_noise_expo(npz_file):
    global noise_data

    return_data, return_label = load_npz_data(npz_file)

    # TODO
    # add noise
    result_data = list()
    result_label = list()
    for i, one_data in enumerate(return_data):

        for one_rate in gv.EXPO_NOISE_RATIO:

            selected_noise_data = random.choice(noise_data)

            selected_noise_data = selected_noise_data*one_rate

            added_data = selected_noise_data + one_data
            added_data = normalization_data(added_data)

            result_data.append(added_data)
            result_label.append(return_label[i])

    result_data = np.array(result_data, dtype=np.float32)
    result_label = np.array(result_label, dtype=np.int8)

    return result_data, result_label




def add_noise_gaussian(npz_file):
    return_data, return_label = load_npz_data(npz_file)

    # TODO
    # add noise
    result_data = list()
    result_label = list()
    for i, one_data in enumerate(return_data):

        for one_rate in gv.GAUSSIAN_NOISE_RATIO:
            gaussian_noise = np.random.normal(0, 1, gv.FULL_SIZE)
            gaussian_noise = normalization_data(gaussian_noise)

            gaussian_noise = gaussian_noise*one_rate

            added_data = gaussian_noise + one_data
            added_data = normalization_data(added_data)

            result_data.append(added_data)
            result_label.append(return_label[i])

    result_data = np.array(result_data, dtype=np.float32)
    result_label = np.array(result_label, dtype=np.int8)

    return result_data, result_label



def load_npz_data(target_npz_file):
    loaded_data = np.load(target_npz_file)
    data = loaded_data['data']
    labels = loaded_data['label']
    return data, labels




def write_npz_file(target_data, target_lable, target_file_path):
    mod_file_path = modify_file_path(target_file_path)

    np.savez(
        mod_file_path,
        data = target_data,
        label = target_lable,
    )
    
    return




def modify_file_path(input_file_path):
    temp = input_file_path.split('/')
    temp_split = temp[-1].split('.')

    filename = temp_split[0] + '_add_zeroth_noise.' + temp_split[-1]

    temp_path = '/'.join(temp[:-1])

    return_path = temp_path + filename

    return return_path




def write_npz_file_gaussian(target_data, target_lable, target_file_path):
    mod_file_path = modify_file_path_gaussian(target_file_path)

    np.savez(
        mod_file_path,
        data = target_data,
        label = target_lable,
    )
    
    return



def modify_file_path_gaussian(input_file_path):
    temp = input_file_path.split('/')
    temp_split = temp[-1].split('.')

    filename = temp_split[0] + '_add_gaussian_noise.' + temp_split[-1]

    temp_path = '/'.join(temp[:-1])

    return_path = temp_path + filename

    return return_path





def draw_single_graph(data):
    fig = plt.figure()
    plt.plot(data)
    plt.show()
    
    return





if __name__ == "__main__":
    print('hello, world~!!')
    gen_noised_data_main()






