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
from tensorflow import keras

import sys
sys.path.append("C:\\Users\\pnclab\\Desktop\\code\\girl_from_ipanema")
from raw_signal_encoder import raw_Encoder_layer


#%%
class cnn_block_1D(layers.Layer):
    
    def __init__(self, **kwarg):
        super(cnn_block_1D, self).__init__()
        
    def __call__(self, inputs, **kwarg):
        x = inputs
        model = keras.Sequential(
            [
                layers.Conv2D(32, (3, 3), padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                layers.Dropout(0.2), 
                
                layers.Conv2D(64, (3, 3), padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                layers.Dropout(0.2), 
                
                layers.Conv2D(128, (3, 3), padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                layers.Dropout(0.2), 
                
                layers.Conv2D(256, (3, 3), padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                layers.Dropout(0.2), 
                ]
            )
        
        x = model(x)
        return x
        
        return



#%%
class cnn_layer_raw(layers.Layer):
    
    def __init__(self, **kwarg):
        super(cnn_layer_raw, self).__init__()
        
                
    def __call__(self, inputs, **kwarg):
        
        if "num_of_classes" in kwarg.keys():
            num_class = kwarg['num_of_classes']
        
        en_layer = raw_Encoder_layer()
        cnn_1D = cnn_block_1D()
        
        x = inputs
        
        x = en_layer(x)
        x = layers.Dropout(0.2)(x)
        x = tf.expand_dims(x, -1)
        x = cnn_1D(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(num_class, activation='softmax')(x)        
        
        return x


#%% __main__
if __name__ == "__main__":
    print("hello, world~!")

