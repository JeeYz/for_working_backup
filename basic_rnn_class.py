# -*- coding: utf-8 -*-

#%% explined
'''
class : basic rnn block
class : normal rnn model
option : last layer is dense layer
'''


#%% declaration
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, GRU
import numpy as np


#%% mormal rnn block
class rnn_block(layers.Layer):
    def __init__(self, **kwarg):
        super(rnn_block, self).__init__()       
        
        if 'num_of_cells' in kwarg.keys():
            self.num_cells = kwarg['num_of_cells']
        if 'rnn_mode' in kwarg.keys():
            self.mode_flag = kwarg['rnn_mode']
            

    def __call__(self, input_val, **kwarg):
        
        if self.mode_flag == 'lstm':
            output_val = LSTM(self.num_cells)(input_val)
            
        elif self.mode_flag == 'gru':
            output_val = GRU(self.num_cells)(input_val)
            
        elif self.mode_flag == 'bilstm':
            output_val = tf.keras.layers.Bidirectional(\
                            LSTM(self.num_cells))(input_val)
            
        elif self.mode_flag == 'bigru':
            output_val = tf.keras.layers.Bidirectional(\
                            GRU(self.num_cells))(input_val)
        
        elif self.mode_flag == 'lstm_reseq':
            output_val = LSTM(self.num_cells, return_sequences=True)(input_val)
            
        elif self.mode_flag == 'gru_reseq':
            output_val = GRU(self.num_cells, return_sequences=True)(input_val)
            
        elif self.mode_flag == 'bilstm_reseq':
            output_val = tf.keras.layers.Bidirectional(LSTM(self.num_cells, 
                                        return_sequences=True))(input_val)
            
        elif self.mode_flag == 'bigru_reseq':
            output_val = tf.keras.layers.Bidirectional(GRU(self.num_cells, 
                                        return_sequences=True))(input_val)
        
        return output_val


#%% class normal rnn
class normal_rnn(layers.Layer):
    def __init__(self, **kwarg):
        super(normal_rnn, self).__init__()
        
        if 'num_of_cells' in kwarg.keys():
            self.num_cells = kwarg['num_of_cells']
        if 'cnn_input' in kwarg.keys():
            self.cnn_input = kwarg['cnn_input']
        if 'num_of_layers' in kwarg.keys():
            self.num_layers = kwarg['num_of_layers']
        if 'rnn_mode' in kwarg.keys():
            self.mode_flag = kwarg['rnn_mode']
        if 'num_of_classes' in kwarg.keys():
            self.num_classes = kwarg['num_of_classes']
        if 'define_dimension' in kwarg.keys():
            self.defined_d = kwarg['define_dimension']
            
    
    def convert_fcnn_trnn(self, input_tensor):
        # input_tensor = input_tensor.numpy()
        # temp = input_tensor.shape
        # result_numpy = np.reshape(input_tensor, (temp[0], temp[1], -1))
        # return result_numpy
        
        print("+++++", input_tensor.shape)
        
        if self.defined_d == '2D':
            result_tensor = tf.reshape(input_tensor, [-1, input_tensor.shape[1], \
                                    input_tensor.shape[2]*input_tensor.shape[3]])
        elif self.defined_d == '1D':
            result_tensor = tf.reshape(input_tensor, [-1, input_tensor.shape[1], \
                                    input_tensor.shape[2]])
        
        print(result_tensor)
        
        return result_tensor
    
    
    def __call__(self, input_data, **kwarg):
        if 'batch_normalization' in kwarg.keys():
            bn_bool = kwarg['batch_normalization']
        if 'num_of_classes' in kwarg.keys():
            num_classes = kwarg['num_of_classes']
            
        print('****', input_data.shape)
            
        if self.cnn_input:
            input_data = self.convert_fcnn_trnn(input_data)
        
        print('****', input_data.shape)
        if self.num_layers > 1:
            rnn_layer_1 = rnn_block(num_of_cells=self.num_cells, 
                                       rnn_mode='bilstm_reseq')
            rnn_layer_2 = rnn_block(num_of_cells=self.num_cells, 
                                       rnn_mode='bilstm_reseq')
            rnn_layer_3 = rnn_block(num_of_cells=self.num_cells, 
                                       rnn_mode='bilstm_reseq')
            rnn_layer_4 = rnn_block(num_of_cells=self.num_cells, 
                                       rnn_mode='bilstm_reseq')
            rnn_layer_5 = rnn_block(num_of_cells=self.num_cells, 
                                       rnn_mode='bilstm_reseq')
            rnn_layer_6 = rnn_block(num_of_cells=self.num_cells, 
                                       rnn_mode='bilstm')
            x = rnn_layer_1(input_data)
            x = layers.Dropout(0.2)(x)
            # x = rnn_layer_2(x)
            # x = rnn_layer_3(x)
            # x = rnn_layer_4(x)
            # x = rnn_layer_5(x)
            x = rnn_layer_6(x)
        else:
            init_rnn_layer = rnn_block(num_of_cells=self.num_cells, 
                                       rnn_mode=self.mode_flag)
            x = init_rnn_layer(input_data)
        
        # x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)
        
        answer = layers.Dense(num_classes, activation='softmax')(x)
                
        return answer



#%% __main__
if __name__ == '__main__':
    print('hello, world~!~!')
    



