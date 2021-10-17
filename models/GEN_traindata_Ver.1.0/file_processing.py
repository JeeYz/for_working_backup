
from global_variables import *


#%%
class File_Processing():
    def __init__(self, **kwargs):
        pass

    
    def run_processing_data(self, filepath):
        
        files_list = self.find_data_files(filepath, '.wav')
        files_list = self.except_bad_data(files_list)
        files_list = self.make_files_dictionary(files_list)
        files_list = self.detect_right_and_wrong(files_list)

        return files_list


    def find_data_files(self, filepath, file_ext):

        all_data_file = list()

        for (path, dir, files) in os.walk(filepath):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == file_ext:
                    file_name = path + "\\" + filename
                    all_data_file.append(file_name)

        return all_data_file


    def except_bad_data(self, files_list):
        result_list = list()

        for one_file in files_list:
            if "kjh" not in one_file:
                result_list.append(one_file)

        return result_list


    def detect_right_and_wrong(self, files_list):

        for one_file in files_list:

            one_label = one_file['label']

            if one_label is not 'none':
                one_file['answer'] = True
            else:
                one_file['answer'] = False

        return files_list


    def make_files_dictionary(self, input_list):
        
        result_list = list()

        for one_filename in input_list:
            one_dict = dict()
            one_dict['filename'] = one_filename
            for one_key in label_dict.keys():
                if one_key in one_filename:
                    # print(str(one_key))
                    one_dict['label'] = one_key
                    break
                else:
                    one_dict['label'] = 'none'
            
            result_list.append(one_dict)
        

        return result_list


    











