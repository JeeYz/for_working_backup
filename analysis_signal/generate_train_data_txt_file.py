


import os
import numpy as np
from numpy.core.numeric import full
from numpy.lib.npyio import load

import matplotlib.pyplot as plt
from scipy.io import wavfile



def draw_multi_graphes(data_list, label_list, row, col):

    fig, ax = plt.subplots( row, col,
                            figsize=(15,12),
                            tight_layout=True)

    data_num = 0

    for i in range(row):
        for j in range(col):
            ax[i, j].plot(data_list[data_num])
            ax[i, j].set_title(str(label_list[data_num]))

            ax[i, j].axhline(y=1.0, color='r', linewidth=1)
            data_num+=1

    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()   
    # plt.tight_layout(pad=0.1)
    return



def draw_single_graph(data, title_name):
    plt.figure()
    plt.plot(data)

    plt.xlabel('sample rate')
    plt.ylabel('amplitude')
    plt.title(title_name)

    plt.tight_layout()
    # plt.show()

    return


def generate_txt_file():

    loaded_data_16000 = np.load("Z:\\02. SW_\\04_TF_모델\\음성\\동기화폴더\\train_data_small_16000_random_.npz")
    data_16000 = loaded_data_16000['data'][1000:2000]
    label_16000 = loaded_data_16000['label'][1000:2000]

    loaded_data_20000 = np.load("Z:\\02. SW_\\04_TF_모델\\음성\\동기화폴더\\train_data_small_20000_random_.npz")
    data_20000 = loaded_data_20000['data'][1000:2000]
    label_20000 = loaded_data_20000['label'][1000:2000]


    f_1 = open("D:\\data_for_train_16000.txt", "w", encoding='utf-8')
    f_2 = open("D:\\data_for_train_20000.txt", "w", encoding='utf-8')


    for i, (p_1, p_2) in enumerate(zip(data_16000, data_20000)):
        for j, (q_1, q_2) in enumerate(zip(p_1, p_2)):
            f_1.write(str(q_1))
            f_1.write("  ")

            f_2.write(str(q_2))
            f_2.write("  ")

        f_1.write(str(label_16000[i]))
        f_1.write("\n")

        f_2.write(str(label_20000[i]))
        f_2.write("\n")

    f_1.close()
    f_2.close()



def read_txt_file():

    full_data_16000 = list()
    full_data_20000 = list()
    full_data_mid = list()

    with open("D:\\data_for_train_16000.txt", "r", encoding='utf-8') as f0,\
        open("D:\\data_for_train_20000.txt", "r", encoding='utf-8') as f1,\
        open("D:\\data_for_train_mid.txt", "r", encoding='utf-8') as f2:

        while True:
            line_0 = f0.readline()
            line_1 = f1.readline() 
            line_2 = f2.readline() 

            line_0 = line_0.split()
            line_1 = line_1.split()
            line_2 = line_2.split()

            if not line_0: break

            new_line_0, new_line_1, new_line_2 = list(), list(), list()

            for i, l in enumerate(line_0):
                new_line_0.append(float(l))
                new_line_1.append(float(line_1[i]))
                new_line_2.append(float(line_2[i]))

            full_data_16000.append(np.asarray(new_line_0[:-1]), dtype=np.float32)
            full_data_20000.append(np.asarray(new_line_1[:-1]), dtype=np.float32)
            full_data_mid.append(np.asarray(new_line_2[:-1]), dtype=np.float32)

            



def divide_data():
    file_0 = "Z:\\02. SW_\\04_TF_모델\\음성\\동기화폴더\\train_data_small_16000_random_.npz"
    file_1 = "Z:\\02. SW_\\04_TF_모델\\음성\\동기화폴더\\train_data_small_20000_random_.npz"

    loaded_0 = np.load(file_0)
    loaded_1 = np.load(file_1)


    data_0, label_0 = list(), list()
    data_1, label_1 = list(), list()

    maximum_data_num = 1000
    minimum_data_num = 100

    none_data_num = 0

    for i, one_data in enumerate(loaded_0["data"]):

        if loaded_0['label'][i] != 5:
            data_0.append(one_data)
            label_0.append(loaded_0['label'][i])
        
        else:
            if none_data_num > minimum_data_num:
                continue
            data_0.append(one_data)
            label_0.append(loaded_0['label'][i])
            none_data_num+=1

        if len(data_0) > maximum_data_num:
            break

    none_data_num = 0

    for i, one_data in enumerate(loaded_1["data"]):

        if loaded_1['label'][i] != 5:
            data_1.append(one_data)
            label_1.append(loaded_1['label'][i])
        
        else:
            if none_data_num > minimum_data_num:
                continue
            data_1.append(one_data)
            label_1.append(loaded_1['label'][i])
            none_data_num+=1

        if len(data_1) > maximum_data_num:
            break


    new_file_0 = "D:\\data_for_train_16000_1000EA.npz"
    new_file_1 = "D:\\data_for_train_20000_1000EA.npz"

    data_0 = np.asarray(data_0, dtype=np.float32)
    data_1 = np.asarray(data_1, dtype=np.float32)

    np.savez(new_file_0, data=data_0, label=label_0)
    np.savez(new_file_1, data=data_1, label=label_1)



def process_for_draw_graph_train_data():

    files_path = "D:\\"
    files_list =    [
                        "data_for_train_16000_1000EA.npz",
                        "data_for_train_20000_1000EA.npz",
                        "data_for_train_mid_1000EA.npz"
                    ]

    data_list_0 = np.load(files_path+files_list[0])["data"]
    label_list_0 = np.load(files_path+files_list[0])["label"]

    data_list_1 = np.load(files_path+files_list[1])["data"]
    label_list_1 = np.load(files_path+files_list[1])["label"]

    data_list_2 = np.load(files_path+files_list[2])["data"]
    label_list_2 = np.load(files_path+files_list[2])["label"]

    row_num = 4
    col_num = 5

    split_data_num = row_num*col_num
    full_data_num = 1000

    for i in range(full_data_num//split_data_num):
        draw_multi_graphes( data_list_0[i*split_data_num:(i+1)*split_data_num], 
                            label_list_0[i*split_data_num:(i+1)*split_data_num],
                            row_num, col_num)
        plt.savefig("D:\\temp_graphes_images\\"+files_list[0]+"_"+str('%02d'%i)+".png", dpi=300)

    for i in range(full_data_num//split_data_num):
        draw_multi_graphes( data_list_1[i*split_data_num:(i+1)*split_data_num], 
                            label_list_1[i*split_data_num:(i+1)*split_data_num],
                            row_num, col_num)
        plt.savefig("D:\\temp_graphes_images\\"+files_list[1]+"_"+str('%02d'%i)+".png", dpi=300)

    for i in range(full_data_num//split_data_num):
        draw_multi_graphes( data_list_2[i*split_data_num:(i+1)*split_data_num], 
                            label_list_2[i*split_data_num:(i+1)*split_data_num],
                            row_num, col_num)
        plt.savefig("D:\\temp_graphes_images\\"+files_list[2]+"_"+str('%02d'%i)+".png", dpi=300)

    return


def search_saturation_data():

    parent_folder = "D:\\voice_data_backup\\PNC_DB_ALL\\PNCDB"
    # parent_folder = "D:\\voice_data_backup\\PNC_DB_ALL"
    file_ext = ".wav"

    command_list = [

        "camera", "end", "picture",
        "stop", "record"

    ]

    files_list = list()
    
    for (path, dir, files) in os.walk(parent_folder):

        print("path : {a}, dir : {b}, files : {c}".format(  a=path, 
                                                            b=dir, 
                                                            c=files ))


        # for filename in files:
        #     ext = os.path.splitext(filename)[-1]
        #     if ext == file_ext:
                
        #         file_name = path + "\\" + filename

        #         plt.figure()
        #         plt.plot(data)

        #         plt.xlabel('sample rate')
        #         plt.ylabel('amplitude')
        #         plt.title(title_name)

        #         plt.tight_layout()

        #         files_list.append("%s\\%s" % (path, filename))



    return










if __name__ == "__main__":
    print("hello, world~!!")
    # process_for_draw_graph_train_data()
    search_saturation_data()

    

    