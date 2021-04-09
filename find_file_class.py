# -*- coding: utf-8 -*-

#%% explaining
'''
class : make files list and text file
    methods :
        1. make label dictionary
        2. read wav files
        3. making file process
        4. write files list in text file
        5. find files recursively
'''


#%% declaration
import os
from scipy.io import wavfile
import numpy as np
import time


#%% class init && make label dict
class make_files_list_asr:
    
    def __init__(self, **kwarg):
        
        if "find_dir_path" in kwarg.keys():
            find_dir_path = kwarg["find_dir_path"]
            self.train_dir_path = find_dir_path + '\\train'
            self.test_dir_path = find_dir_path + '\\test'
        if "file_format" in kwarg.keys():
            self.file_format = kwarg["file_format"]
        if "return_file_path" in kwarg.keys():
            self.make_file_path = kwarg["return_file_path"]
        if "label_file_path" in kwarg.keys():
            temp = kwarg["label_file_path"]
        
        self.train_files_list = []
        self.test_files_list = []
        self.label_dict = dict()
        self.make_dict_labels(temp)
        

#%% old version    
    def make_label_dict(self, t_path): # t_path is label file's path
        none_list = ['hibixbi', 'hiplus', 'okgoogle']
        label_num = 0
        with open(t_path, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:break
                line = line.split()
                
                if line[0] not in none_list:
                    self.label_dict[line[0]] = label_num
                    label_num += 1 
                    
                    
#%% new version
    def make_dict_labels(self, path):
        self.label_dict=dict()
        num = 0
        with open(path, 'r', encoding='utf-8') as rf:
            while True:
                line = rf.readline()
                line = line.split()
                if not line: break
                self.label_dict[line[0]]=num
                num+=1
            self.label_dict['hipnc2'] = 20
    
     
#%% read wav file and make numpy binary file
    def read_wav_file(self, **kwarg):
        
        if "numpy_data_filename" in kwarg.keys():
            result_binary = kwarg["numpy_data_filename"]
        if "text_file_list" in kwarg.keys():
            text_filename = kwarg["text_file_list"]
        if "mode_flag" in kwarg.keys():
            mode_flag = kwarg['mode_flag']        
        
        temp = self.make_file_path +'\\'+ text_filename
        temp_1 = self.make_file_path +'\\'+ result_binary
        
        data_list = list()
        label_list = list()
        sr_list = list()
        
        with open(temp, 'r', encoding='utf-8') as fr,\
        open(temp_1, 'wb') as fwb:
            while True:
                line = fr.readline()
                line = line.split()
                if not line: break

                fs, data = wavfile.read(line[0])
                
                if mode_flag == 1:
                    a = int(line[1])
                elif mode_flag == 2:
                    a = int(line[2])
                
                data_list.append(data)
                label_list.append(a)
                sr_list.append(fs)
                
            data_list = np.asarray(data_list)
            label_list = np.asarray(label_list)
            sr_list = np.asarray(sr_list)
            np.savez_compressed(fwb, label=label_list, data=data_list, rate=sr_list)
                
        return
    
    
#%% modify text file with adding lables
    def modify_text_file_add_label(self):
        file1 = 'train_text.txt'
        file2 = 'test_text.txt'
        file3 = 'train_text_labels.txt'
        file4 = 'test_text_labels.txt'    
        anti_keywords = ['hibixbi', 'hiplus', 'okgoogle']
        keywords = ['hipnc', 'hipnc2']
        label_file = 'label_list.txt'
                    
        label_dict = self.make_dict_labels(label_file)
        
        with open(file1, 'r', encoding='utf-8') as f1, open(file3, 'w', encoding='utf-8') as f2:
            while True:
                line = f1.readline()
                if not line: break
                line_t = line.split("\\")
                line = line.split("\n")
                
                if line_t[0] != '':
                    if line_t[-2] in keywords:
                        mode_1_num = 1
                    else:
                        mode_1_num = 0
                    
                    if line_t[-2] in anti_keywords or line_t[-2] == 'none':
                        mode_2_num = 0
                    elif line_t[-2] in keywords:
                        mode_2_num = 0
                    else:
                        mode_2_num = label_dict[line_t[-2]]
                
                new_line = line[0]+'\t\t'+str(mode_1_num)+'\t\t'+str(mode_2_num)+'\n'
                f2.write(new_line)
        
        with open(file2, 'r', encoding='utf-8') as f1, open(file4, 'w', encoding='utf-8') as f2:
            while True:
                line = f1.readline()
                if not line: break
                line_t = line.split("\\")
                line = line.split("\n")
                
                if line_t[0] != '':
                    if line_t[-2] in keywords:
                        mode_1_num = 1
                    else:
                        mode_1_num = 0
                    
                    if line_t[-2] in anti_keywords or line_t[-2] == 'none':
                        mode_2_num = 0
                    elif line_t[-2] in keywords:
                        mode_2_num = 0
                    else:
                        mode_2_num = label_dict[line_t[-2]]
                
                new_line = line[0]+'\t\t'+str(mode_1_num)+'\t\t'+str(mode_2_num)+'\n'
                f2.write(new_line)
        
        return

    
    
#%% # this func is for making files list
    def make_file_process(self):     
        self.find_target_files()
        self.make_files_list_file()
        

#%% make file of files list
    def make_files_list_file(self):
        train_text_filename = 'train_text.txt'
        test_text_filename = 'test_text.txt'
        
        temp = self.make_file_path +'\\'+ train_text_filename
        with open(temp, 'w', encoding='utf-8') as wf:
            for one_file in self.train_files_list:
                wf.write(one_file)
                wf.write("\n")
                
        temp = self.make_file_path +'\\'+ test_text_filename
        with open(temp, 'w', encoding='utf-8') as wf:
            for one_file in self.test_files_list:
                wf.write(one_file)
                wf.write("\n")
        
        return
        

#%% find files in recursive mechanism
    def find_target_files(self):

        for (path, dir, files) in os.walk(self.train_dir_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == self.file_format:
                    self.train_files_list.append(path+'\\'+filename)
        
        for (path, dir, files) in os.walk(self.test_dir_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == self.file_format:
                    self.test_files_list.append(path+'\\'+filename)


#%% find sub dics and files
    def return_directories_name(self):
        
        Target_list = os.listdir(self.path)
        Target_list_data = [file for file in Target_list if file.endswith(self.file_format)]
        
        return Target_list_data


#%% __main__
if __name__ == '__main__':
    print("hello, world~!")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    