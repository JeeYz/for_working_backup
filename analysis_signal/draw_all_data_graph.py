
import numpy as np
import matplotlib.pyplot as plt


train_npz_file_path = "D:\\train_data_mini_20000_random_.npz"
# train_npz_file_path = "D:\\train_data_test_20000_random_.npz"
test_npz_file_path = "D:\\test_data_20000_.npz"




def draw_graphes(cl_train_data, cl_test_data):
    
    plt_graph_save_path = "D:\\temp_graphes_images\\"
    plt_graph_filename = "all_train_and_test_data_"

    col_num = 7
    row_num = 7

    file_num = 0
    data_num = 0

    while data_num < len(cl_train_data):

        fig, axs = plt.subplots(    7, 7,
                                    figsize=(10, 10), 
                                    constrained_layout=True
                                )

        for n in range(col_num):
            for m in range(row_num):
                if data_num >= len(cl_train_data):
                    break
                axs[n, m].plot(cl_train_data[data_num])
                data_num += 1
                print("{}th file is done...{}".format(file_num+1, data_num), end='\r')
                if data_num >= len(cl_train_data):
                    break

        filename_str = plt_graph_save_path + plt_graph_filename + str('%02d'%file_num) + ".png"
        plt.savefig(filename_str, dpi=300)
        file_num += 1


    # data_num = 0

    # while data_num < len(cl_test_data):

    #     fig, axs = plt.subplots(    7, 7,
    #                                 figsize=(10, 10), 
    #                                 constrained_layout=True
    #                             )

    #     for n in range(col_num):
    #         for m in range(row_num):
    #             axs[n, m].plot(cl_test_data[data_num])
    #             data_num += 1
    #             if data_num >= len(cl_test_data):
    #                 break

    #     filename_str = plt_graph_save_path + plt_graph_filename + str('%02d'%file_num) + ".png"
    #     plt.savefig(filename_str, dpi=300)
    #     file_num += 1


    return



def classify_data(train_load, test_load):

    cl_train_data = list()
    cl_test_data = list()

    train_data = train_load["data"]
    train_label = train_load["label"]

    test_data = test_load["data"]
    test_label = test_load["label"]

    for i, l in enumerate(train_label):
        if l != 5:
            cl_train_data.append(train_data[i])
    
    for i, l in enumerate(test_label):
        if l != 5:
            cl_test_data.append(test_data[i])

    return cl_train_data, cl_test_data



def load_npz_data():

    train_load = np.load(train_npz_file_path)
    test_load = np.load(test_npz_file_path)

    return train_load, test_load



def load_one_npz_data():

    test_load = np.load(test_npz_file_path)

    cl_test_data = list()

    test_data = test_load["data"]
    test_label = test_load["label"]

    for i, l in enumerate(test_label):
        if l != 5:
            cl_test_data.append(test_data[i])

    return cl_test_data




def main():

    # train_load, test_load = load_npz_data()
    # cl_train_data, cl_test_data = classify_data(train_load, test_load)

    temp = list()
    loaded_data = load_one_npz_data()
    draw_graphes(loaded_data, temp)



    return



if __name__ == "__main__":
    print("hello, world~!!")
    main()


