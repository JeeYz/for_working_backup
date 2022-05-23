import global_variables as gv
from files_module import find_data_files as fdf

import numpy as np
import random


def add_noise_expo(input_data):
    noise_data = gen_expo_noise_block()

    result = input_data + noise_data

    return result



def gen_expo_noise_block():
    noise_files_list = fdf(gv.noise_data_path, ".wav")



    return result




if __name__ == "__main__":
    print('hello, world~!!')






