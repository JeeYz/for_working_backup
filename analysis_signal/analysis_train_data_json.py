
import os
import numpy as np


from numpy.core.numeric import full
from numpy.lib.npyio import load

import matplotlib.pyplot as plt
from scipy.io import wavfile

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import json



plt_graph_save_path = "D:\\temp_graphes_images\\"
plt_graph_filename = "compare_norm_satu_"
file_num = 0

max_len = 0


def draw_data_graph(whole_data):

    global file_num, max_len

    for one_data in whole_data:

        # 1번 그래프에 사용
        _, norm_raw_sig = wavfile.read(one_data["normal_data"])
        _, satu_raw_sig = wavfile.read(one_data["saturation_data"])

        if len(norm_raw_sig) > len(satu_raw_sig):
            gap_num = len(norm_raw_sig)-len(satu_raw_sig)
            satu_raw_sig = np.pad(satu_raw_sig, gap_num, 'constant', constant_values=(0))
        elif len(norm_raw_sig) < len(satu_raw_sig):
            gap_num = len(satu_raw_sig)-len(norm_raw_sig)
            norm_raw_sig = np.pad(norm_raw_sig, gap_num, 'constant', constant_values=(0))

        norm_stft_data = return_stft_data(norm_raw_sig)
        satu_stft_data = return_stft_data(satu_raw_sig)

        # 2번 그래프에 사용
        norm_stft_data_T_log = np.log(np.transpose(norm_stft_data))
        satu_stft_data_T_log = np.log(np.transpose(satu_stft_data))

        norm_resize_data = return_resizing_data(norm_stft_data)
        satu_resize_data = return_resizing_data(satu_stft_data)

        # 3번 그래프에 사용
        norm_resize_data_T = np.transpose(norm_resize_data)
        satu_resize_data_T = np.transpose(satu_resize_data)

        # 4번 그래프에 사용
        norm_resize_data_T_log = np.log(norm_resize_data_T)
        satu_resize_data_T_log = np.log(satu_resize_data_T)
        


        fig, axs = plt.subplots(    4, 2,
                                    figsize=(6, 10), 
                                    constrained_layout=True
                                )

        title_str = "speaker: {}, command: {}".format(one_data["speaker"], one_data["command"])
        fig.suptitle(title_str, fontsize=16)

        # 정상 그래프
        axs[0, 0].plot(norm_raw_sig)
        axs[0, 0].set_title("normal data")

        axs[1, 0].pcolor(norm_stft_data_T_log)
        axs[1, 0].set_title("stft T+log")

        axs[2, 0].pcolor(norm_resize_data_T)
        axs[2, 0].set_title("resize T")

        axs[3, 0].pcolor(norm_resize_data_T_log)
        axs[3, 0].set_title("resize T+log")

        # 포화 그래프
        axs[0, 1].plot(satu_raw_sig)
        axs[0, 1].set_title("saturation data")

        axs[1, 1].pcolor(satu_stft_data_T_log)
        axs[1, 1].set_title("stft T+log")

        axs[2, 1].pcolor(satu_resize_data_T)
        axs[2, 1].set_title("resize T")

        axs[3, 1].pcolor(satu_resize_data_T_log)
        axs[3, 1].set_title("resize T+log")


        filename_str = plt_graph_save_path + plt_graph_filename + str('%02d'%file_num) + ".png"
        plt.savefig(filename_str, dpi=300)
        file_num+=1
    # plt.show()

    return




def return_resizing_data(data):

    data = np.expand_dims(data, axis=-1)
    result = preprocessing.Resizing(32, 32)(data)
    result = np.squeeze(result, axis=-1)

    return result




def return_stft_data(data):

    data = np.array(data, dtype=np.float32)
    x = tf.signal.stft(data, frame_length=255, frame_step=128)
    result = tf.abs(x)

    return result




def load_json_data(satu_index_num):

    json_file = "temp.json"

    whole_data = list()

    with open(json_file, "r") as jf:
        loaded_data = json.load(jf)

    

    for one_speaker in loaded_data["speakers"]:

        temp = dict()
        for one_command in one_speaker["commands"]:

            satu_num = len(one_command["saturation_data"])
            norm_num = len(one_command["normal_data"])

            if satu_num > 2 and \
                    norm_num > 2:
                temp["speaker"] = one_speaker["name"]
                temp["command"] = one_command["command"]
                temp["saturation_data"] = one_command["saturation_data"][satu_index_num]
                temp["normal_data"] = one_command["normal_data"][0]

                whole_data.append(temp)


    print(len(whole_data))

    return whole_data




def load_json_data_saturation_only():

    json_file = "temp.json"

    whole_data = list()

    with open(json_file, "r") as jf:
        loaded_data = json.load(jf)

    

    for one_speaker in loaded_data["speakers"]:

        temp = dict()
        for one_command in one_speaker["commands"]:

            satu_num = len(one_command["saturation_data"])
            norm_num = len(one_command["normal_data"])

            if one_command["saturation_data"] == []:
                    continue

            if norm_num == 0:
                    # and\
                    # satu_num != 0:

                temp["speaker"] = one_speaker["name"]
                temp["command"] = one_command["command"]
                temp["saturation_data"] = one_command["saturation_data"][0]

                whole_data.append(temp)


    print(len(whole_data))

    # for w in whole_data:
    #     print(w)

    return whole_data



def load_json_data_normal_only():

    json_file = "temp.json"

    whole_data = list()

    with open(json_file, "r") as jf:
        loaded_data = json.load(jf)


    for one_speaker in loaded_data["speakers"]:

        temp = dict()
        for one_command in one_speaker["commands"]:

            satu_num = len(one_command["saturation_data"])
            norm_num = len(one_command["normal_data"])

            if one_command["normal_data"] == []:
                    continue

            if satu_num == 0:
                    # and\
                    # satu_num != 0:

                temp["speaker"] = one_speaker["name"]
                temp["command"] = one_command["command"]
                temp["normal_data"] = one_command["normal_data"][0]

                whole_data.append(temp)


    print(len(whole_data))

    # for w in whole_data:
    #     print(w)

    return whole_data



def draw_data_graph_satu_only(whole_data):

    global file_num, max_len

    for one_data in whole_data:

        # 1번 그래프에 사용
        _, satu_raw_sig = wavfile.read(one_data["saturation_data"])

        gap_num = (42000-len(satu_raw_sig))//2

        # print(len(satu_raw_sig), gap_num)
        satu_raw_sig = np.pad(satu_raw_sig, (gap_num, 42000-gap_num-len(satu_raw_sig)), 'constant', constant_values=(0))

        satu_stft_data = return_stft_data(satu_raw_sig)

        # 2번 그래프에 사용
        satu_stft_data_T_log = np.log(np.transpose(satu_stft_data))

        satu_resize_data = return_resizing_data(satu_stft_data)

        # 3번 그래프에 사용
        satu_resize_data_T = np.transpose(satu_resize_data)

        # 4번 그래프에 사용
        satu_resize_data_T_log = np.log(satu_resize_data_T)
        


        fig, axs = plt.subplots(    4, 1,
                                    figsize=(5, 10), 
                                    constrained_layout=True
                                )

        title_str = "speaker: {}, command: {}".format(one_data["speaker"], one_data["command"])
        fig.suptitle(title_str, fontsize=16)

        # 포화 그래프
        axs[0].plot(satu_raw_sig)
        axs[0].set_title("saturation data")

        axs[1].pcolor(satu_stft_data_T_log)
        axs[1].set_title("stft T+log")

        axs[2].pcolor(satu_resize_data_T)
        axs[2].set_title("resize T")

        axs[3].pcolor(satu_resize_data_T_log)
        axs[3].set_title("resize T+log")


        filename_str = plt_graph_save_path + plt_graph_filename + str('%02d'%file_num) + ".png"
        plt.savefig(filename_str, dpi=300)
        file_num+=1
    # plt.show()

    return




def draw_data_graph_norm_only(whole_data):

    global file_num, max_len

    for one_data in whole_data:

        # 1번 그래프에 사용
        _, satu_raw_sig = wavfile.read(one_data["normal_data"])

        gap_num = (42000-len(satu_raw_sig))//2


        # print(len(satu_raw_sig), gap_num)
        satu_raw_sig = np.pad(satu_raw_sig, (gap_num, 42000-gap_num-len(satu_raw_sig)), 'constant', constant_values=(0))

        satu_stft_data = return_stft_data(satu_raw_sig)

        # 2번 그래프에 사용
        satu_stft_data_T_log = np.log(np.transpose(satu_stft_data))

        satu_resize_data = return_resizing_data(satu_stft_data)

        # 3번 그래프에 사용
        satu_resize_data_T = np.transpose(satu_resize_data)

        # 4번 그래프에 사용
        satu_resize_data_T_log = np.log(satu_resize_data_T)
        


        fig, axs = plt.subplots(    4, 1,
                                    figsize=(5, 10), 
                                    constrained_layout=True
                                )

        title_str = "speaker: {}, command: {}".format(one_data["speaker"], one_data["command"])
        fig.suptitle(title_str, fontsize=16)

        # 포화 그래프
        axs[0].plot(satu_raw_sig)
        axs[0].set_title("normal data")

        axs[1].pcolor(satu_stft_data_T_log)
        axs[1].set_title("stft T+log")

        axs[2].pcolor(satu_resize_data_T)
        axs[2].set_title("resize T")

        axs[3].pcolor(satu_resize_data_T_log)
        axs[3].set_title("resize T+log")


        filename_str = plt_graph_save_path + plt_graph_filename + str('%02d'%file_num) + ".png"
        plt.savefig(filename_str, dpi=300)
        file_num+=1
    # plt.show()

    return




def temp_fuction():

    json_file = "temp.json"

    whole_data = list()

    with open(json_file, "r") as jf:
        loaded_data = json.load(jf)


    for one_speaker in loaded_data["speakers"]:

        temp = dict()
        for one_command in one_speaker["commands"]:

            satu_num = len(one_command["saturation_data"])
            norm_num = len(one_command["normal_data"])

            # if one_command["normal_data"] != [] and\
            #         one_command["command"] == "camera":

            #     temp["speaker"] = one_speaker["name"]
            #     temp["command"] = one_command["command"]
            #     temp["normal_data"] = one_command["normal_data"][0]

            #     whole_data.append(temp)

            if one_command["saturation_data"] != [] and\
                    one_command["command"] == "picture":

                temp["speaker"] = one_speaker["name"]
                temp["command"] = one_command["command"]
                temp["normal_data"] = one_command["saturation_data"][0]

                whole_data.append(temp)




    print(len(whole_data))

    # for w in whole_data:
    #     print(w)


    return whole_data







def main():

    global max_len

    # w_data = load_json_data(0)
    # draw_data_graph(w_data)

    # w_data = load_json_data_saturation_only()
    # draw_data_graph_satu_only(w_data)

    # w_data = load_json_data_normal_only()
    # draw_data_graph_norm_only(w_data)

    w_data = temp_fuction()
    draw_data_graph_norm_only(w_data)

    return






if __name__ =='__main__':
    print("hello, world~!!")
    main()




