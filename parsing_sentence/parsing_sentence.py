
import os
from unittest import result

target_path = "D:\\voice_data_backup\\zeroth_korean.tar\\zeroth_korean\\"

result_file = "D:\\result_words_CW_2022.txt"

cond0 = ["은", "는", "이", "가", "을", "를"]
cond1 = ["은", "는"]
cond2 = ["들"]
cond3 = ["하는"]
cond4 = ["했", "겠", "셨", "하다", "였", "씨", "졌", "없", "갔", "투데", "었", "났", "가까", "것"]

cond_ex = ["이", "가"]


def write_result_file(input_list):

    with open(result_file, "w", encoding='utf-8') as wf:
        for one_word in input_list:
            wf.write(one_word[:-1])
            wf.write("\n")
    
    return


def filter_of_words(input_list):
    result = list()
    
    for one_word in input_list:
        if len(one_word) > 2:
            if cond2[0] != one_word[-2]:
                temp_flag = 0
                for one_cond4 in cond4:
                    if one_cond4 in one_word:
                        temp_flag = 0
                        break
                    else:
                        temp_flag = 1

                if temp_flag == 1:
                    result.append(one_word)

    return result


def read_single_sent(target_txt_file):
    result = list()
    with open(target_txt_file, "r", encoding='utf-8') as txtfile:
        while True:
            line = txtfile.readline()
            if not line: break
            line = line.split()

            for one_word in line[1:]:
                for one_cond in cond_ex:
                    if one_cond == one_word[-1]:
                        if len(one_word[:-1]) > 1:
                            result.append(one_word)
            
    return result


def find_text_files():
    result = list()
    file_ext = ".txt"

    for (path, dir, files) in os.walk(target_path):

        for filename in files:

            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                target_filename = path + "\\" + filename
                result.append(target_filename)
    
    return result


def main():

    temp = list()
    final_result = list()
    result = find_text_files()

    for one_file in result:
        # print(one_file)
        for one_result in read_single_sent(one_file):
            temp.append(one_result)

    final_result = filter_of_words(temp)

    final_result = list(set(final_result))
    final_result.sort()

    print(final_result)
    print(len(final_result))

    write_result_file(final_result)

    return



if __name__ == "__main__":
    print("hello, world~!!")
    main()
    
    