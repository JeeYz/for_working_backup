
import sys
import random
import os
import threading

temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)


pcm_path = 'D:\\voice_data_backup\\AI_HUB_data_speech'
pcm_text_list = 'D:\\voice_data_backup'


##
def write_text_file(files_list):
    with open(pcm_text_list+'\\all_text.txt', 'w', encoding='utf-8') as f:
        for i,one_file in enumerate(files_list):
            with open(one_file, 'r') as fr:
                while True:
                    # print(one_file)
                    line = fr.readline()
                    if not line: break

                    f.write(line)

            print("\rone file is done... %d" %(i+1), end='')

    return


##
def find_files(filepath, file_ext):
    files_list = list()
    for (path, dir, files) in os.walk(filepath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                # print("%s\\%s" % (path, filename))
                files_list.append("%s\\%s" % (path, filename))

    return files_list



##
def main():
    pcm_files_list = list()
    pcm_files_list = find_files(pcm_path, '.txt')
    # print(pcm_files_list)
    print(len(pcm_files_list))
    write_text_file(pcm_files_list)
    return





if __name__ == '__main__':
    main()

## endl
