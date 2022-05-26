
from global_variables import *
import global_variables as gv

from trigger_algorithm import normalization_for_data 
import files_module as fm
from add_noise import normalization_data

from python_speech_features import mfcc






def gen_VAD_data_main():
    
    loaded_data = np.load("/home/pncdl/DeepLearning/CWtraindata/npzTrain/speaker_59183529_cmd_F_result_file_traindata.npz")
    
    data_list = loaded_data['data']

    for one_data in data_list:

        fm.draw_single_graph(one_data)

        start_time = time.time()

        # x = librosa.feature.melspectrogram(y=one_data, sr=16000, n_mels=128, fmax=8000)
        
        x = librosa.feature.mfcc(y=one_data, sr=16000, n_mfcc=40)
        # x = librosa.feature.mfcc(S=x)

        computing_time = time.time() - start_time

        print(computing_time)

        fm.draw_mfcc_graph(x)

        start_time = time.time()

        x = mfcc(one_data, numcep=100)

        computing_time = time.time() - start_time

        print(computing_time)

        x = transpose_the_matrix(x)

        fm.draw_mfcc_graph(x)

        print('\n')

    
    return



def transpose_the_matrix(data):
    return np.swapaxes(data, 0, 1)






if  __name__ == '__main__':
    print('hello, world~!')
    gen_VAD_data_main()




