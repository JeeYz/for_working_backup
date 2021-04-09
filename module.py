# -*- coding: utf-8 -*-

import numpy as np
import os
from datetime import datetime

#%% return max number
def return_max_number(*args, **kwarg):
    max_num = 0
    for i, n in enumerate(args):
        if i == 0:
            max_num = n
        else:
            if n > max_num:
                max_num = n
    
    return max_num


#%% find max num
## find max number in numpy mfcc feature matrix data
def find_max_number_in_mfcc(*args):
    max_n = 0
    for one_data in args:
        for one_el in one_data:
            if len(one_el) > max_n:
                max_n = len(one_el)
            
    return max_n


#%% class save the best weights and model
class save_the_best:
    def __init__(self, **kwarg):
        if 'file_name' in kwarg.keys():
            self.filename = kwarg['file_name']
        
        self.max_metrics = 0.
        if os.path.isfile(self.filename):
            self.max_metrics = self.load_best_metrics()
        else:
            with open(self.filename, 'w', encoding='utf-8') as fw:
                now = datetime.now()
                fw.write(str(now)+'####')
                fw.write(str(0.0)+'####accuracy')
        
    def load_best_metrics(self):
        with open(self.filename, 'r', encoding='utf-8') as fr:
            line = fr.readline()
            line = line.split('####')
            self.max_metrics = float(line[1])
    
    def write_best_metrics(self, **kwarg):
        if 'kind_of_metric' in kwarg.keys():
            metr = kwarg['kind_of_metric']
        if 'metric_result' in kwarg.keys():
            self.max_metrics = kwarg['metric_result']
        
        now = datetime.now()
        with open('highest_metrics.txt', 'w', encoding='utf-8') as fw:
            fw.write(str(now)+'####')
            fw.write(str(self.max_metrics)+'####'+metr)
            
    def initialization_max(self):
        now = datetime.now()
        with open('highest_metrics.txt', 'w', encoding='utf-8') as fw:
            fw.write(str(now)+'####')
            fw.write(str(0.0)+'####accuracy')
        
    
    
#%% __main__
if __name__ == '__main__':
    print('hello, world~!!')
        
    
    
    
    
    
    
    
    