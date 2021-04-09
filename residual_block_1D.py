# -*- coding: utf-8 -*-

#%% explained
'''
class : residual cnn block
class : convolution nn with using resnet
'''


#%% declaration
from tensorflow.keras import layers
import tensorflow as tf


#%% class -> residual cnn block
class residual_cnn_block_1D(layers.Layer):
    
    def __init__(self, **kwarg):
        super(residual_cnn_block_1D, self).__init__()
          
        if "channel_size" in kwarg.keys():
            self.chan_size = kwarg['channel_size']
        if "kernel_size" in kwarg.keys():
            self.ker_size = kwarg['kernel_size']
        if "strides_size" in kwarg.keys():
            self.strides_size = kwarg['strides_size']
      
        
    def __call__(self, init_inputs, inputs, **kwarg):
        conv2d_layer_1 = layers.Conv1D(self.chan_size[0], self.ker_size, 
                                       strides=self.strides_size,
                                       padding='same')
        conv2d_layer_2 = layers.Conv1D(self.chan_size[1], self.ker_size, 
                                       strides=self.strides_size,
                                       padding='same')
        
        init_val = inputs
        
        x = conv2d_layer_1(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = conv2d_layer_2(x)
        print('*********', self.chan_size)
        y = layers.Conv1D(self.chan_size[1], 1, padding='same')(init_val)
        z = layers.Conv1D(self.chan_size[1], 1, padding='same')(init_inputs)
        x = tf.math.add(y, x)
        x = tf.math.add(x, z)
        x = tf.nn.relu(x)
                
        return x
        

#%% class -> residual net
class residual_net_1D(layers.Layer):
    
    def __init__(self, **kwarg):
        super(residual_net_1D, self).__init__()
        
        if "num_of_blocks" in kwarg.keys():
            self.num_of_blocks = kwarg["num_of_blocks"]
        if "pooling_bool" in kwarg.keys():
            self.pooling_bool = kwarg["pooling_bool"]
        else:
            self.pooling_bool = False
        if "kernel_size" in kwarg.keys():
            self.ker_size = kwarg["kernel_size"]
        if "strides_size" in kwarg.keys():
            self.strides_size = kwarg['strides_size']
        if "output_type" in kwarg.keys():
            # 1: 'cnn' / 2: 'rnn' / 3: 'dense' / 4: 'false'
            self.output_type = kwarg['output_type'] 

        
        self.residual_cnn_layer_1 = residual_cnn_block_1D(channel_size=[8, 8], 
                            strides_size=self.strides_size, kernel_size=self.ker_size)
        self.residual_cnn_layer_2 = residual_cnn_block_1D(channel_size=[16, 16], 
                            strides_size=self.strides_size, kernel_size=self.ker_size)
        # self.residual_cnn_layer_3 = residual_cnn_block_1D(channel_size=[256, 256], 
        #                     strides_size=self.strides_size, kernel_size=self.ker_size)
        # self.residual_cnn_layer_4 = residual_cnn_block_1D(channel_size=[512, 512], 
        #                     strides_size=self.strides_size, kernel_size=self.ker_size)
        # self.residual_cnn_layer_5 = residual_cnn_block_1D(channel_size=[128, 128], 
        #                     strides_size=self.strides_size, kernel_size=self.ker_size)
        # self.residual_cnn_layer_6 = residual_cnn_block_1D(channel_size=[256, 256], 
        #                     strides_size=self.strides_size, kernel_size=self.ker_size)
            
        # self.pooling_layer = layers.MaxPool1D(pool_size=4, padding='same')
        self.pooling_layer = layers.MaxPool1D(pool_size=4, padding='same')
        
        
    def __call__(self, inputs, **kwarg):
        
        if 'num_of_classes' in kwarg.keys():
            num_class = kwarg['num_of_classes']        
        if 'dense_softmax' in kwarg.keys():
            softmax_bool = kwarg['dense_softmax']
        else:
            softmax_bool = False
        
        x = layers.Conv1D(8, 50, strides=30, padding='same')(inputs)
        init_input = x
        
        x = self.residual_cnn_layer_1(init_input, x)
        x = self.residual_cnn_layer_2(init_input, x)
        # x = self.residual_cnn_layer_3(init_input, x)
        # x = self.residual_cnn_layer_4(init_input, x)
        # x = self.residual_cnn_layer_5(init_input, x)
        # x = self.residual_cnn_layer_6(init_input, x)
        
        
        if softmax_bool:
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dropout(0.1)(x)
            x = layers.Dense(128, activation='relu')(x)
            output_val = layers.Dense(num_class, activation='softmax')(x)
        else:
            x = layers.Conv1D(256, 50, strides=30, padding='same')(x)
            # x = layers.Conv1D(256, 5, strides=3, padding='same')(x)
            # x = self.pooling_layer(x)
            # x = self.pooling_layer(x)
            output_val = x
        
        return output_val


#%% __main__
if __name__ == "__main__":
    print("hello, world~!")

