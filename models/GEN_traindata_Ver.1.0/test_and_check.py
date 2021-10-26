

from global_variables import *
import modifying_data_and_info as moddata

def check_number_of_labels(files_list):

    temp_list = list()
    for one_file in files_list:
        one_label = one_file['label']
        if one_label not in temp_list: 
            temp_list.append(one_label)

    temp_dict = dict()
    for one in temp_list:
        temp_dict[one] = 0
    
    for one_file in files_list:
        one_label = one_file['label']
        temp_dict[one_label]+=1

    print("**numbers of labels**")
    for one in temp_dict.keys():
        print("{label} : {num}".format(label=one, num=temp_dict[one]))
    
    print("\n")

    return


def check_data_gap_size(input_data_list):

    under_full_size_num  = 0
    over_full_size_num = 0

    print_result_list = list()

    for one_file in input_data_list:
        for one_data in one_file['data']:
            if one_data['gap_start_end'] <= FULL_SIZE:
                under_full_size_num += 1
            else:
                over_full_size_num += 1
                if one_file not in print_result_list:
                    print_result_list.append(one_file)


    print("***number of data size***")
    print("number of under full size : {under}".format(under=under_full_size_num))
    print("number of over full size : {over}".format(over=over_full_size_num))
    # print(print_result_list)

    result_dict = dict()
    for one_file in print_result_list:
        temp_label_name = one_file['label']
        for one_data in one_file['data']:
            if one_file['label'] in result_dict.keys():
                result_dict[temp_label_name] += 1
            else:
                result_dict[temp_label_name] = 0

    print("\n")
    print("*number of data over full size for each labels")
    for one_label in result_dict.keys():
        print("{label} : {data_num}".format(label=one_label, data_num=result_dict[one_label]))

    print('\n')

    return



def check_data_length(input_data_list):

    for one_file in input_data_list:
        for one_data in one_file['data']:
            temp_length = one_data['data_length']
            if temp_length != FULL_SIZE:
                try:
                    raise Exception("데이터의 길이가 full size와 일치하지 않습니다.")
                except Exception as e:
                    print(e)
                    print("filename : {filename}, length : {length}".format(
                        filename=one_file['filename'],
                        length=temp_length,
                    ))

                    if len(one_data['data']) != FULL_SIZE:
                        try:
                            raise Exception("실 데이터의 길이의 오류 확인")
                        except Exception as e:
                            print(e)
                            result_data = moddata.fit_full_size_data(one_data['data'])
                            one_data['data'] = result_data
                            one_data['data_length'] = len(result_data)
                            print("수정 완료...")
                    else:
                        one_data['data_length'] = FULL_SIZE
                        print("데이터 길이 값 수정 완료...")

                    if len(one_data['data']) != FULL_SIZE:
                        raise Exception("오류 재확인 수정 코드를 수정 필요")
                    else:
                        print("재확인 이상 없음...")       
                

    return



def check_traindata_npz_with_plt(filename):

    loaded_data = np.load(filename)

    target_data = loaded_data['data']
    target_label = loaded_data['label']

    # print(target_data.shape, target_label.shape)

    whole_data = zip(target_data, target_label)

    data_list = list()

    for one_data in whole_data:
        data_list.append(one_data)

        if len(data_list) is 30:
            draw_multi_graphes(data_list)
            data_list = list()


    return


def draw_multi_graphes(input_data_list):
    num_data = len(input_data_list)
    x_num = int(np.sqrt(num_data))

    for i in range(sys.maxsize):
        if num_data <= x_num*i:
            y_num = i
            break

    fig, axs = plt.subplots(x_num, y_num)

    for i, ax in enumerate(axs.flat):
        # print(input_data_list[i])
        if input_data_list[i][0] is None:
        # if input_data_list[i] is None:
            break
        x = np.linspace(0, FULL_SIZE, num=FULL_SIZE)
        # y = input_data_list[i][0]
        y = input_data_list[i]
        ax.plot(x, y)
        ax.set_title(input_data_list[i][1])
        ax.grid()

    figManager = plt.get_current_fig_manager()
    # plt.tight_layout()
    figManager.window.showMaximized()

    plt.show()

    return


def draw_single_graph(input_data):
    fig = plt.figure()
    plt.plot(input_data)
    plt.show()
    return




if __name__ == '__main__':
    print('hello, world~!!')
    check_traindata_npz_with_plt(numpy_traindata_files_path)



