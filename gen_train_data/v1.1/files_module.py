from global_variables import *
import global_variables as gv

def find_data_files( filepath, file_ext):

    all_data_file = list()
    print("finding target is {path}, {ext}...".format(path=filepath, ext=file_ext))

    for (path, dir, files) in os.walk(filepath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_ext:
                file_name = path + "/" + filename
                all_data_file.append(file_name)

    return all_data_file


def draw_single_graph(data):
    fig = plt.figure()
    plt.plot(data)
    plt.show()



def draw_mfcc_graph(data):

    plt.figure()
    plt.pcolormesh(data)

    plt.xlabel('frame sequence')
    plt.ylabel('number of filters')

    plt.colorbar()
    plt.show()

    return




