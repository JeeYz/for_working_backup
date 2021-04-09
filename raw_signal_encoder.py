# -*- coding: utf-8 -*-
#%% explained
'''
class : basic cnn block
class : normal cnn model
dense of softmax : optional
'''


#%% declaration
from tensorflow.keras import layers
import tensorflow as tf


#%%
class raw_Encoder_layer_block(layers.Layer):
    
    def __init__(self, **kwarg):
        super(raw_Encoder_layer_block, self).__init__()
        if "ker_size" in kwarg.keys():
            self.ker_size = kwarg['ker_size']
        if "stride_size" in kwarg.keys():
            self.st_size = kwarg['stride_size']
        if "channel_size" in kwarg.keys():
            self.ch_size = kwarg['channel_size']
                
    def __call__(self, inputs, **kwarg):
        
        x = inputs
        x = layers.Conv1D(self.ch_size, self.ker_size, 
                          strides=self.st_size, padding='same')(x)
        x = layers.MaxPooling1D(4)(x)
        x = layers.BatchNormalization()(x)
        
        return x


#%%
class raw_Encoder_layer(layers.Layer):

    def __init__(self, **kwarg):
        super(raw_Encoder_layer, self).__init__()
        if "layers_depth" in kwarg.keys():
            self.layers_dep = kwarg['layers_depth']
                
    def __call__(self, inputs, **kwarg):
        
        x = inputs
        
        block_0 = raw_Encoder_layer_block(ker_size=4, stride_size=1, channel_size=32)
        block_1 = raw_Encoder_layer_block(ker_size=4, stride_size=1, channel_size=64)
        block_2 = raw_Encoder_layer_block(ker_size=4, stride_size=1, channel_size=128)
        block_3 = raw_Encoder_layer_block(ker_size=4, stride_size=1, channel_size=256)
    
        x = block_0(x)
        x = block_1(x)
        x = block_2(x)
        x = block_3(x)
        
        return x
    
    

#%% __main__
if __name__ == "__main__":
    print("hello, world~!")

