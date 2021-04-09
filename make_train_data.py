# -*- coding: utf-8 -*-

#%% explained
'''
class : load train data class
    method : 
        1. load train data from npz
        2. This  making for padding data 'post'
        3. option : 
            1. We can decide 
'''


#%% declaration
import tensorflow as tf
import numpy as np
from module import find_max_number_in_mfcc as find_max

#%% class for loading data
class load_train_data_class:
    def __init__(self, **kwarg):
        if 'files_list' in kwarg.keys():
            self.files_list = kwarg['files_list']
        if 'npz_path' in kwarg.keys():
            self.npz_path = kwarg['npz_path']
                    
    def load_train_data(self, *args, **kwarg):
        
        if 'load_train_kind' in kwarg.keys():
            load_dataset = kwarg['load_train_kind']
        if 'max_number' in kwarg.keys():
            max_num = kwarg['max_number']
        
        train_load_data = np.load(self.npz_path+self.files_list[args[0]], allow_pickle=True)
        test_load_data = np.load(self.npz_path+self.files_list[args[1]], allow_pickle=True)
        
        train_labels = train_load_data['label']
        train_feats = train_load_data['data']
        train_rates = train_load_data['rate']
        
        test_labels = test_load_data['label']
        test_feats = test_load_data['data']
        test_rates = test_load_data['rate']
          
        max_number = find_max(train_feats, test_feats)
        # max_number = 512
        
        train_feats = tf.keras.preprocessing.sequence.pad_sequences(train_feats, 
                                                maxlen=max_number, padding='post', dtype='float32')
        test_feats = tf.keras.preprocessing.sequence.pad_sequences(test_feats, 
                                                maxlen=max_number, padding='post', dtype='float32')
        
        if load_dataset == 'cnn':
            train_feats = tf.expand_dims(train_feats, -1)
            print("data shape : "+ str(train_feats.shape))
            test_feats = tf.expand_dims(test_feats, -1)
            print("data shape : "+ str(test_feats.shape))
            
            conv_shape = (train_feats.shape[1], train_feats.shape[2], 1)
            
            return train_feats, test_feats, train_labels, test_labels, conv_shape
        
        elif load_dataset == 'rnn':
            print("data shape : "+ str(train_feats.shape))
            input_shape = (train_feats.shape[1], train_feats.shape[2])
            print('###', input_shape)
            return train_feats, test_feats, train_labels, test_labels, input_shape
        
        elif load_dataset == 'raw_signal':
            # train_feats = tf.keras.preprocessing.sequence.pad_sequences(train_feats, 
            #                                     maxlen=43*768, padding='post', dtype='float32')
            # test_feats = tf.keras.preprocessing.sequence.pad_sequences(test_feats, 
            #                                         maxlen=43*768, padding='post', dtype='float32')
            train_feats = tf.expand_dims(train_feats, -1)
            # train_feats = tf.expand_dims(train_feats, -1)
            print("data shape : "+ str(train_feats.shape))
            test_feats = tf.expand_dims(test_feats, -1)
            # test_feats = tf.expand_dims(test_feats, -1)
            print("data shape : "+ str(test_feats.shape))
            
            # train_feats = tf.reshape(train_feats, [-1, 43, 768])
            # train_feats = train_feats.reshape(-1, 43, 768)
            # test_feats = test_feats.reshape(-1, 43, 768)
            
            # conv_shape = (train_feats.shape[1], train_feats.shape[2], 1)
            conv_shape = (train_feats.shape[1], train_feats.shape[2])
            
            return train_feats, test_feats, train_labels, test_labels, conv_shape


#%% __main__
if __name__ == '__main__':
    print('hello, world~!!')
