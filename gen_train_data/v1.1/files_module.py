from global_variables import *
import global_variables as gv

def find_data_files( filepath, file_ext):

    all_data_file = list()
    print("This function is called for finding data files...")

    for (path, dir, files) in os.walk(filepath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                file_name = path + "\\" + filename
                all_data_file.append(file_name)

    return all_data_file
