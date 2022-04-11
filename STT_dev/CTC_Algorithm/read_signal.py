import numpy as np
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

data_path = "D:\\temp\\AMRFiles"
file_ext = ".wav"


def find_data_files( filepath, file_ext):

    all_data_file = list()
    print("This function is called for finding data files...")

    for (path, dir, files) in os.walk(filepath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                file_name = path + "\\" + filename
                all_data_file.append(file_name)

    return all_data_file


def draw_graph_raw_signal(data, **kwargs):

    title_name = kwargs['title_name']

    plt.figure()
    plt.plot(data)

    plt.xlabel('sample rate')
    plt.ylabel('amplitude')
    plt.title(title_name)

    # plt.tight_layout()
    plt.show()

    return


def standardization_func(data):
    return (data-np.mean(data))/np.std(data)


def main():

    files_list = find_data_files(data_path, file_ext)

    for one_file in files_list: 
        sampler, data = wavfile.read(one_file)
        data = standardization_func(data)
        draw_graph_raw_signal(data, title_name=one_file)

    return

    
if __name__ == "__main__":
    main()


