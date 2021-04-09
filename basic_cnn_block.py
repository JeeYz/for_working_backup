# -*- coding: utf-8 -*-
#%% explained
'''
class : basic cnn block
class : normal cnn model
dense of softmax : optional
'''


#%% declaration
from tensorflow.keras import layers


#%% class -> cnn_block
class cnn_block(layers.Layer):
    
    def __init__(self, **kwarg):
        super(cnn_block, self).__init__()
        
        if "channel_size" in kwarg.keys():
            self.chan_size = kwarg["channel_size"]
        
    def __call__(self, inputs, **kwarg):
        
        conv2d_layer_1 = layers.Conv2D(self.chan_size, (3, 3),
                                       padding='same', activation='relu')
        conv2d_layer_2 = layers.Conv2D(self.chan_size, (3, 3), 
                                       padding='same', activation='relu')
        
        x = conv2d_layer_1(inputs)
        x = layers.BatchNormalization()(x)
        x = conv2d_layer_2(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.MaxPooling2D(pool_size=(3, 1), padding='same')(x)
        
        output_ts = layers.Dropout(0.2)(x)
                
        return output_ts
        

#%% class -> norm_cnn
class norm_cnn(layers.Layer):
    
    def __init__(self, **kwarg):
        super(norm_cnn, self).__init__()
        
        self.conv_block_0 = cnn_block(channel_size=32)
        self.conv_block_1 = cnn_block(channel_size=64)
        self.conv_block_2 = cnn_block(channel_size=128)
        
        
        self.conv_layer_0 = layers.Conv2D(256, (3,3), padding='same',
                                          activation='relu')
        self.conv_layer_1 = layers.Conv2D(256, (3,3), padding='same',
                                          activation='relu')
        
    def __call__(self, inputs, **kwarg):
        
        if 'num_of_classes' in kwarg.keys():
            num_class = kwarg['num_of_classes']        
        
        x = self.conv_block_0(inputs)        
        x = self.conv_block_1(x)        
        x = self.conv_block_2(x)
        
        x = self.conv_layer_0(x)
        x = layers.BatchNormalization()(x)
        x = self.conv_layer_1(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Flatten()(x)
        # x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='linear')(x)
        # x = layers.Dropout(0.3)(x)
        output_val = layers.Dense(num_class, activation='softmax')(x)        
        
        return output_val


#%% __main__
if __name__ == "__main__":
    print("hello, world~!")

