from global_variables import *
import random


white_noise_group = [
    '59183640',
    '59184297',
    '59184387',
    '59185145',
    '59185238',
    '59185713',
    '59184211',
    '59184290',
]

device_noise_group = [
    # '59184088',
    # '59185312',
    # '59185353',
    # '59185622',
    '59185653',
    # '59185031',
    # '59184205',
    # '59184752',
]

electric_noise_group = [
    '59185990',
    '59198984',
]

saturation_group = [
    '59185228',
    '59192927',
]

all_groups = [
    white_noise_group,
    device_noise_group,
    electric_noise_group,
    saturation_group,
]


def return_data(input_filepath):

    try:
        with wave.open(input_filepath, 'rb') as wf:
            parameters = wf.getparams()
    except FileNotFoundError as e:
        print(e)

    a = wavio.read(input_filepath)

    data0 = a.data[:, 0]
    curr_data = data0 

    return curr_data


def draw_single_graph(input_data, title_name):
    fig = plt.figure()
    plt.plot(input_data)
    plt.title(title_name)
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    plt.show()
    return


def return_random(total_num, persentage_num):
    persentage_num = persentage_num*10
    num = random.randrange(total_num)
    if num < persentage_num:
        return 1
    else:
        return 0


def return_file_dict(filename):
    temp_departures = filename.split('\\')
    temp_filename = temp_departures[-1]

    temp_dict = dict()
    temp_dict['filename'] = temp_filename
    temp_dict['filepath'] = filename
    return temp_dict


def return_one_dict(input_list, filename):
    temp_departures = filename.split('\\')
    temp_filename = temp_departures[-1]
    temp_speaker = temp_filename.split('_')[0]
    temp_sort = temp_departures[-2]

    temp_file_dict = return_file_dict(filename)

    input_flag = -1

    if input_list == []:
        temp_dict = dict()
        temp_dict['speaker'] = temp_speaker
        temp_dict['sort'] = temp_sort

        if 'files' not in temp_dict.keys():
            temp_dict['files'] = list()

        temp_dict['files'].append(temp_file_dict)
        input_list.append(temp_dict)

    else:
        for one_dict in input_list:
            if one_dict['speaker'] == temp_speaker:
                input_flag = 1
                break
            else:
                input_flag = 0
            
    if input_flag == 1:
        for one_dict in input_list:
            if one_dict['speaker'] == temp_speaker:
                one_dict['files'].append(temp_file_dict)
                break

    elif input_flag == 0:
        temp_dict = dict()
        temp_dict['speaker'] = temp_speaker
        temp_dict['sort'] = temp_sort

        if 'files' not in temp_dict.keys():
            temp_dict['files'] = list()

        temp_dict['files'].append(temp_file_dict)
        input_list.append(temp_dict)

    return input_list


def return_top_dict():
    temp_dict = dict()
    temp_dict['white_noise'] = list()
    temp_dict['device_noise'] = list()
    temp_dict['electric_noise'] = list()
    temp_dict['saturation'] = list()
    return temp_dict





if __name__ == '__main__':
    print('hello, world~!!')

    data_path = "C:\\temp"
    files_list = fpro.find_data_files(data_path, ".wav")

    whole_data_dict = return_top_dict()
    keys_list = list(whole_data_dict.keys())

    for i,one_group in enumerate(all_groups):
        for one_speaker in one_group:
            for one_file in files_list:
                if one_speaker in one_file:
                    temp = whole_data_dict[keys_list[i]]
                    temp_list = return_one_dict(
                        temp,
                        one_file,
                    )
                    temp = temp_list

    # print(json.dumps(whole_data_dict, indent=4, ensure_ascii=False))


    for one_key in keys_list:
        temp_data = whole_data_dict[one_key]
        for one_speaker in temp_data:
            for one_file in one_speaker['files']:
                curr_data = return_data(one_file['filepath'])
                title_name = one_key+' '+one_file['filename']
                draw_single_graph(curr_data, title_name)











