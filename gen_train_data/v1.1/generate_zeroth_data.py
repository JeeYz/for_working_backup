from global_variables import *
import global_variables as gv
from trigger_algorithm import normalization_for_data 


flac_target_path = '/home/pncdl/DeepLearning/CWtraindata/zeroth_korean'
finding_ext = '.flac'
target_ext = ".wav"
data_len_threshold = 10000
full_zeroth_data_npz_limit = 100
curr_num_of_files = 0

target_npz_files_path = "/home/pncdl/DeepLearning/CWtraindata/npzTrain/"
npz_files_name_header = 'zeroth_none_data_'


# convert flac files to wav files
def os_system_command_ffmpeg_for_converting(input_file):
    temp = input_file.split('/')
    temp_filename = temp[-1].split('.')[0]
    target_filename = temp_filename + target_ext
    target_file = "/".join(temp[:-1]) + "/" + target_filename
    
    command_line = 'ffmpeg -i ' + input_file + " " + target_file 
    os.system(command_line)

    return




def convert_flac_to_wav():
    result_list = fm.find_data_files(flac_target_path, finding_ext)
    print(result_list)

    for one_file in result_list:
        os_system_command_ffmpeg_for_converting(one_file)
        print(one_file, end='\r')

    return




# generate npz files for train
def generate_zeroth_main():
    global curr_num_of_files

    zeroth_files_list = fm.find_data_files(flac_target_path, target_ext)
    print(len(zeroth_files_list))

    zeroth_data_list = list()    
    
    for one_file in zeroth_files_list:
        sr, data = wavfile.read(one_file)
        data = np.array(data, dtype=gv.TRAIN_DATA_TYPE)
        return_result_list_of_data = cut_signal_data(data)
        zeroth_data_list.extend(return_result_list_of_data)

        if len(zeroth_data_list) >= data_len_threshold:

            gen_npz_files(zeroth_data_list)

            zeroth_data_list = list()
            curr_num_of_files += 1


        if full_zeroth_data_npz_limit < curr_num_of_files:
            print("here is break...")
            break

    regenerate_json_file()

    return




def cut_signal_data(input_data):

    data_length = len(input_data)
    num_of_blocks = data_length//gv.FULL_SIZE

    result = list()

    for i in range(num_of_blocks+1):
        if i >= num_of_blocks:

            temp_data = input_data[i*gv.FULL_SIZE:]

            if len(temp_data) < gv.FULL_SIZE:
                zero_padded_data = add_zero_padding(temp_data)
                temp_data = zero_padded_data[:gv.FULL_SIZE]
            if len(temp_data) > gv.FULL_SIZE:
                temp_data = input_data[i*gv.FULL_SIZE:(i+1)*gv.FULL_SIZE]

            temp_data = np.array(temp_data, dtype=np.float32)
            temp_data = normalization_for_data(temp_data)

            if str(np.max(temp_data)) == 'nan':
                print(np.max(temp_data))
            else:
                result.append(temp_data)

        else:
            temp_data = input_data[i*gv.FULL_SIZE:(i+1)*gv.FULL_SIZE]

            temp_data = np.array(temp_data, dtype=np.float32)
            temp_data = normalization_for_data(temp_data)

            if str(np.max(temp_data)) == 'nan':
                print(np.max(temp_data))
            else:
                result.append(temp_data)

    return result




def add_zero_padding(input_data):
    zero_padding = np.zeros(gv.FULL_SIZE, dtype=gv.TRAIN_DATA_TYPE)
    result = np.append(input_data, zero_padding)
    return result 




def gen_npz_files(zeroth_data_list):
    global curr_num_of_files

    data_list = list()
    label_list = list()

    for one_data in zeroth_data_list:
        data_list.append(one_data)
        label_list.append(0)

    data_list = np.array(data_list, dtype=gv.TRAIN_DATA_TYPE)
    label_list = np.array(label_list, dtype=np.int8)

    target_npz_path = target_npz_files_path + npz_files_name_header + str(curr_num_of_files).zfill(3) + '.npz'

    np.savez(
        target_npz_path,
        data = data_list,
        label = label_list,
    )
    
    return




def regenerate_json_file():
    npz_files_list = fm.find_data_files(gv.npz_target_path, '.npz')

    json_file_path = gv.CWdata_path + gv.npz_data_json_filename 

    with open(json_file_path, 'w', encoding='utf-8') as fw:
        json.dump(
            npz_files_list, 
            fw, 
            indent='\t', 
            ensure_ascii=False
        )

    return
















if __name__ == '__main__':
    print('hello, world~!')
    generate_zeroth_main()
    


