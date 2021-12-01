

from pickle import load
from global_variables import *
import modifying_data_and_info as moddata
import global_variables as gv
import random


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
        for one_data in one_file['file_data']:
            if one_data['gap_start_end'] <= gv.FULL_SIZE:
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
        for one_data in one_file['file_data']:
            temp_length = one_data['data_length']
            if temp_length != gv.FULL_SIZE:
                try:
                    raise Exception("데이터의 길이가 full size와 일치하지 않습니다.")
                except Exception as e:
                    print(e)
                    print("filename : {filename}, length : {length}".format(
                        filename=one_file['filename'],
                        length=temp_length,
                    ))

                    if len(one_data['data']) != gv.FULL_SIZE:
                        try:
                            raise Exception("실 데이터의 길이의 오류 확인")
                        except Exception as e:
                            print(e)
                            result_data = moddata.fit_full_size_data(one_data['data'])
                            one_data['data'] = result_data
                            one_data['data_length'] = len(result_data)
                            print("수정 완료...")
                    else:
                        one_data['data_length'] = gv.FULL_SIZE
                        print("데이터 길이 값 수정 완료...")

                    if len(one_data['data']) != gv.FULL_SIZE:
                        raise Exception("오류 재확인 수정 코드를 수정 필요")
                    else:
                        print("재확인 이상 없음...")       
                

    return



def check_traindata_npz_with_plt(filename):
    print(filename)

    loaded_data = np.load(filename)

    target_data = loaded_data['data']
    target_label = loaded_data['label']

    # print(target_data.shape, target_label.shape)

    whole_data = zip(target_data, target_label)

    data_list = list()

    for one_data in whole_data:
        data_list.append(one_data)

        if len(data_list) is NUMBER_OF_GRAPH:
            draw_multi_graphes(data_list)
            data_list = list()

    draw_multi_graphes(data_list)

    return


def draw_multi_graphes(input_data_list):
    length_data = len(input_data_list[0][0])
    num_data = len(input_data_list)
    x_num = int(np.sqrt(num_data))

    for i in range(sys.maxsize):
        if num_data <= x_num*i:
            y_num = i
            break

    fig, axs = plt.subplots(x_num, y_num)

    # print(num_data) 
    for i, ax in enumerate(axs.flat):
        # print(input_data_list[i])
        # print(i, ax)
        # if input_data_list[i][0] is None:
        # # if input_data_list[i] is None:
        #     break
        if i >= num_data:
            break
        x = np.linspace(0, length_data, num=length_data)
        y = input_data_list[i][0]
        # y = input_data_list[i]
        ax.plot(x, y)
        ax.set_title(input_data_list[i][1])
        ax.grid()

    figManager = plt.get_current_fig_manager()
    # plt.tight_layout()
    figManager.window.showMaximized()

    plt.show()

    return


def draw_single_graph(input_data, titlename):
    fig = plt.figure()
    plt.plot(input_data)
    plt.title(titlename)
    plt.show()
    return


def return_random(total_num, persentage_num):
    persentage_num = persentage_num*10
    num = random.randrange(total_num)
    if num < persentage_num:
        return 1
    else:
        return 0


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as fr:
        loaded_data = json.load(fr)

    ######################################
    temp = list()
    for one_file in loaded_data:
        if 'noncmd' in one_file:
            temp.append(one_file)
    
    loaded_data = temp
    ######################################

    for one_file in loaded_data:
        check_traindata_npz_with_plt(one_file)

    # temp = speaker_exception.keys()
    # print(temp)

    # target_list = list()
    # target_key = 'none'

    # for one_file in loaded_data:
    #     for one_speaker in speaker_exception[target_key]:
    #         if one_speaker in one_file:
    #             target_list.append(one_file)
    
    # for one_file in target_list:
    #     check_traindata_npz_with_plt(one_file)

    
def check_hy_cw_data():
    hy_data_path = "D:\\voice_data_backup\\PNC_DB_ALL\\PNCDB"
    cw_data_path = "C:\\temp\\PnC_Solution_CW_all_1102\\"
    
    hy_files_list = fpro.find_data_files(hy_data_path, '.wav')
    cw_files_list = fpro.find_data_files(cw_data_path, '.wav')
    
    # for one_file in hy_files_list:
    #      sr, data = wavfile.read(one_file)
    #      draw_single_graph(data, one_file)

         
    for one_file in cw_files_list:
        temp = dict()
        temp['filename'] = one_file
        temp['label'] = 'None'
        result = cwsig.gen_sig_data_2(temp)
        draw_single_graph(result['file_data'], one_file.split('\\')[-1])



if __name__ == '__main__':
    print('hello, world~!!')
    check_hy_cw_data()
        
    
    #################################################################
    # target_json = CWdata_path+'\\'+'$$npz_data.json'
    # load_json_file(target_json)

    # target_path = "C:\\temp\\test_Ver.2.4.npz"
    # check_traindata_npz_with_plt(target_path)

    #################################################################
    # target_path = 'C:\\temp\\test_npz_data\\'
    # target_path = "C:\\temp\\CW_test\\"
    # files_list = fpro.find_data_files(target_path, '.npz')

    # temp = list()
    # for one_file in files_list:
    #     if 'noncmd' in one_file:
    #         temp.append(one_file)

    # files_list = temp

    # for one in files_list:
    #     print(one)
    #     check_traindata_npz_with_plt(one)
        # loaded = np.load(one)
        # data = loaded['data']
        # label = loaded['label']
        # draw_single_graph(data)





