import pyaudioconvert as pac
import os
from scipy.io import wavfile



origin_path = "/home/pncdl/DeepLearning/CWtraindata/PnC_Solution_CW_all_1102/"
target_path = "/home/pncdl/DeepLearning/CWtraindata/PnC_Solution_CW_all_1102_16k/"


mod_keyword = "_16k"
target_ext = ".wav"





def find_data_files( filepath, file_ext):

    all_data_file = list()
    print("This function is called for finding data files...")

    for (path, dir, files) in os.walk(filepath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                file_name = path + "/" + filename
                all_data_file.append(file_name)

    return all_data_file




def convert_24bit_to_16bit(origin_file, target_file):
    pac.convert_wav_to_16bit_mono(origin_file, target_file)
    return




def create_folder(directory_name):
    try:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
    except Exception as e:
        print("Error : {dir_n}을 생성합니다.".fotmat(dir_n=directory_name))
    return




def convert_main():

    wav_files_list = find_data_files(origin_path, target_ext)

    for i, one_file in enumerate(wav_files_list):
        temp_filename = one_file.split('/')
        temp_filename[-4] = temp_filename[-4] + mod_keyword
        new_path = "/".join(temp_filename[:-1])

        create_folder(new_path)

        temp_div_filename = temp_filename[-1].split('.')
        temp_div_filename[0] = temp_div_filename[0] + mod_keyword
        new_filename = ".".join(temp_div_filename)

        new_full_filename = new_path + '/' + new_filename

        convert_24bit_to_16bit(one_file, new_full_filename)

        print("{fname}th file is done".format(fname=(i+1)), end='\r')

    return







if __name__ == '__main__':
    print('hello, world~!!')
    convert_main()


