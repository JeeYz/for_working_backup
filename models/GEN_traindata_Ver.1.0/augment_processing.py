
from global_variables import *


class Augment_Process():
    def __init__(self):
        pass


    def time_stretch_process(self, files_list):

        for one_file in files_list:
            data = copy.deepcopy(one_file['data'][0])
            for rate in rate_list:
                aug_data = librosa.effects.time_stretch(data, rate)
                one_file['data'].append(aug_data)

        return files_list









