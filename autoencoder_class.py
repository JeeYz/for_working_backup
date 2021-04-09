# -*- coding: utf-8 -*-

#%% explanation
'''
auto encoder class...
'''

#%% declaration
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, GRU
import numpy as np


#%% class auto encoder
class AutoEncoder(layers.Layer):
    def __init__(self, **kwarg):
        super(AutoEncoder, self).__init__()
        if 'channel_size' in kwarg.keys():
            channel_size = kwarg['channel_size']
        if 'kernel_size' in kwarg.keys():
            kernel_size = kwarg['kernel_size']
        if 'stride_size' in kwarg.keys():
            stride_size = kwarg['stride_size']
            
        self.encoding_layer = ae_Encoder(channel_size=channel_size, kernel_size=kernel_size,
                                         stride_size=stride_size)
        self.decoding_layer = ae_Decoder(channel_size=channel_size, kernel_size=kernel_size,
                                         stride_size=stride_size)
        
    def __call__(self, input_x):
        x = self.encoding_layer(input_x)
        res = self.decoding_layer(x)
        
        return res


#%% encoder
class ae_Encoder(layers.Layer):
    def __init__(self, **kwarg):
        super(ae_Encoder, self).__init__()
        if 'channel_size' in kwarg.keys():
            channel_size = kwarg['channel_size']
        if 'kernel_size' in kwarg.keys():
            kernel_size = kwarg['kernel_size']
        if 'stride_size' in kwarg.keys():
            stride_size = kwarg['stride_size']
        
        self.conv2D_layer_1 = layers.Conv2D(channel_size[0], kernel_size, activation='relu', 
                                            padding='same', strides=stride_size)
        self.conv2D_layer_2 = layers.Conv2D(channel_size[1], kernel_size, activation='relu', 
                                            padding='same', strides=stride_size)
        self.conv2D_layer_3 = layers.Conv2D(channel_size[2], kernel_size, activation='relu',
                                            padding='same', strides=stride_size)
        self.pooling_layer = layers.MaxPooling2D(pool_size=(4, 1), padding='same')
        
        
    def __call__(self, input_x):
        x = self.conv2D_layer_1(input_x)
        x = self.conv2D_layer_2(x)
        x = self.conv2D_layer_3(x)
        
        return x
        
        
#%% decoder
class ae_Decoder(layers.Layer):
    def __init__(self, **kwarg):
        super(ae_Decoder, self).__init__()
        if 'channel_size' in kwarg.keys():
            channel_size = kwarg['channel_size']
        if 'kernel_size' in kwarg.keys():
            kernel_size = kwarg['kernel_size']
            self.ker = kernel_size
        if 'stride_size' in kwarg.keys():
            stride_size = kwarg['stride_size']
        
        self.conv2D_layer_1 = layers.Conv2DTranspose(channel_size[2], kernel_size[0], activation='relu',
                                            padding='same', strides=(stride_size,1))
        self.conv2D_layer_2 = layers.Conv2DTranspose(channel_size[1], kernel_size[0], activation='relu',
                                            padding='same', strides=(stride_size,1))
        self.conv2D_layer_3 = layers.Conv2DTranspose(channel_size[0], kernel_size[0], activation='relu',
                                            padding='same', strides=(stride_size,1))
        self.upsampling_layer = layers.UpSampling2D(size=(4, 1))
    
    def __call__(self, input_x):
        x = self.conv2D_layer_1(input_x)
        x = self.conv2D_layer_2(x)
        # x = self.conv2D_layer_2(input_x)
        x = self.conv2D_layer_3(x)
        x = layers.Conv2D(1, self.ker, padding='same', activation='sigmoid')(x)
        
        return x


#%% __main__
if __name__ == '__main__':
    print('hello, world~!')