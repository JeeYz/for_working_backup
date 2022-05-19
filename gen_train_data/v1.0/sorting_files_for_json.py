
from global_variables import *


class SortFiles():
    def __init__(self, files_list):
        self.__files_dict = self.initiate_dict(files_list)

    def initiate_dict(self, files_list):
        result = dict()
        result['title'] = "Crowd Works Train Data"
        result['files'] = list()

        for one_file in files_list:
            parsing_temp = one_file.split('\\')
            last_path_temp = '\\'.join(parsing_temp[:-1])
            filename_temp = parsing_temp[-1]
            speaker_temp = filename_temp.split('_')[0]

            temp_dict = dict()
            temp_dict['speaker'] = speaker_temp
            temp_dict['last_path'] = last_path_temp
            temp_dict['filename'] = filename_temp
            temp_dict['origin_file'] = one_file
            temp_dict['target_dir'] = last_path_temp+'\\'+speaker_temp+'\\'
            temp_dict['target_file'] = temp_dict['target_dir']+filename_temp
            
            result['files'].append(temp_dict)

        return result

    def get_files_dict(self) :
        return self.__files_dict

    def set_files_dict(self):
        return

    def print_files_dict(self):
        print(json.dumps(
                        self.__files_dict,
                        sort_keys=False, 
                        indent=4, 
                        default=str, 
                        ensure_ascii=False
        ))

    def move_files(self):
        temp_dict = self.get_files_dict()
        
        for one_dict in temp_dict['files']:
            origin_file = one_dict['origin_file']
            target_file = one_dict['target_file']
            try:
                shutil.move(origin_file, target_file)
            except:
                print("폴더가 존재하지 않습니다. 새로운 폴더를 생성합니다.")
                print(one_dict['target_dir'])
                self.create_folder(one_dict['target_dir'])
                shutil.move(origin_file, target_file)

    def create_folder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)




if __name__ == '__main__':
    print('hello, world~!!')

    files_list = fpro.find_data_files(CWdata_path, '.wav')

    # print(files_list)
    print(len(files_list))

    json_class = SortFiles(files_list)
    # json_class.print_files_dict()

    json_class.move_files()




