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
class residual_cnn_block_2D(layers.Layer):

    def __init__(self, **kwarg):
        super(residual_cnn_block_2D, self).__init__()

        if "channel_size" in kwarg.keys():
            self.chan_size = kwarg['channel_size']


    def __call__(self, inputs, **kwarg):
        conv2d_layer_1 = layers.Conv2D(self.chan_size[0], (3, 3),
                                       padding='same')
        conv2d_layer_2 = layers.Conv2D(self.chan_size[1], (3, 3),
                                       padding='same')

        init_val = inputs
        # inputs = layers.BatchNormalization()(inputs)

        x = conv2d_layer_1(inputs)
        # x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = conv2d_layer_2(x)
        # x = layers.BatchNormalization()(x)
        print('*********', self.chan_size)
        y = layers.Conv2D(self.chan_size[1], (1, 1), padding='same')(init_val)
        x = tf.math.add(y, x)
        x = tf.nn.relu(x)
        # x = layers.MaxPooling2D(pool_size=(2, 1), padding='same')(x)
        x = layers.Dropout(0.2)(x)

        return x


#%% class -> residual net
class residual_net_2D(layers.Layer):

    def __init__(self, **kwarg):
        super(residual_net_2D, self).__init__()

        if "num_of_blocks" in kwarg.keys():
            self.num_of_blocks = kwarg["num_of_blocks"]
        if "pooling_bool" in kwarg.keys():
            self.pooling_bool = kwarg["pooling_bool"]
        else:
            self.pooling_bool = False
        if "output_type" in kwarg.keys():
            # 1: 'cnn' / 2: 'rnn' / 3: 'dense' / 4: 'false'
            self.output_type = kwarg['output_type']
        if 'init_channels' in kwarg.keys():
            self.init_ch = kwarg['init_channels']


        self.residual_cnn_layer_1 = residual_cnn_block_2D(channel_size=[8, 8])
        self.residual_cnn_layer_1_0 = residual_cnn_block_2D(channel_size=[8, 8])
        self.residual_cnn_layer_2 = residual_cnn_block_2D(channel_size=[16, 16])
        self.residual_cnn_layer_2_0 = residual_cnn_block_2D(channel_size=[16, 16])
        self.residual_cnn_layer_3 = residual_cnn_block_2D(channel_size=[32, 32])
        self.residual_cnn_layer_3_0 = residual_cnn_block_2D(channel_size=[32, 32])
        self.residual_cnn_layer_4 = residual_cnn_block_2D(channel_size=[64, 64])
        self.residual_cnn_layer_4_0 = residual_cnn_block_2D(channel_size=[64, 64])
        self.residual_cnn_layer_5 = residual_cnn_block_2D(channel_size=[128, 128])
        self.residual_cnn_layer_5_0 = residual_cnn_block_2D(channel_size=[128, 128])
        self.residual_cnn_layer_6 = residual_cnn_block_2D(channel_size=[256, 256])
        self.residual_cnn_layer_6_0 = residual_cnn_block_2D(channel_size=[256, 256])
        self.residual_cnn_layer_7 = residual_cnn_block_2D(channel_size=[512, 512])
        self.residual_cnn_layer_7_0 = residual_cnn_block_2D(channel_size=[512, 512])
        self.residual_cnn_layer_8 = residual_cnn_block_2D(channel_size=[1024, 1024])

        # self.pooling_layer = layers.MaxPool1D(pool_size=4, padding='same')
        self.pooling_layer = layers.MaxPooling2D(pool_size=(3, 1), padding='same')


    def __call__(self, inputs, **kwarg):

        if 'num_of_classes' in kwarg.keys():
            num_class = kwarg['num_of_classes']
        if 'dense_softmax' in kwarg.keys():
            softmax_bool = kwarg['dense_softmax']
        else:
            softmax_bool = False

        pooling_size = (2, 2)

        # x = layers.Conv1D(self.init_ch, 50, strides=30, padding='same')(inputs)

        # temp = tf.shape(x)
        # x = tf.reshape(x, (-1, temp[1], self.init_ch))
        # x = tf.expand_dims(x, -1)

        # init_input = x

        # x = layers.Conv2D(8, (3, 3), padding='same')(inputs)
        # x = layers.Conv2D(8, (3, 3), padding='same')(x)
        # x = layers.Conv2D(16, (3, 3), padding='same')(x)
        # x = layers.Conv2D(16, (3, 3), padding='same')(x)
        # x = layers.Conv2D(32, (3, 3), padding='same')(x)
        # x = layers.Conv2D(32, (3, 3), padding='same')(x)

        # x = self.residual_cnn_layer_3(x)
        x = self.residual_cnn_layer_3(inputs)
        # x = self.residual_cnn_layer_3_0(x)
        x = layers.MaxPooling2D(pool_size=pooling_size, padding='same')(x)
        # x = layers.BatchNormalization()(x)

        x = self.residual_cnn_layer_4(x)
        # x = self.residual_cnn_layer_3_0(x)
        x = layers.MaxPooling2D(pool_size=pooling_size, padding='same')(x)
        # x = layers.BatchNormalization()(x)

        x = self.residual_cnn_layer_5(x)
        # x = self.residual_cnn_layer_4_0(x)
        x = layers.MaxPooling2D(pool_size=pooling_size, padding='same')(x)

        x = self.residual_cnn_layer_6(x)
        # x = self.residual_cnn_layer_4_0(x)
        x = layers.MaxPooling2D(pool_size=pooling_size, padding='same')(x)

        x = self.residual_cnn_layer_7(x)
        # x = self.residual_cnn_layer_5_0(x)
        x = layers.MaxPooling2D(pool_size=(2, 1), padding='same')(x)

        # x = self.residual_cnn_layer_8(x)

        # x = layers.Conv2D(256, (3, 1), padding='same')(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)



        # x = layers.AveragePooling2D(pool_size=pooling_size, padding='same')(x)


        if softmax_bool:
            x = layers.GlobalAveragePooling2D()(x)
            # x = layers.Conv2D(8, (3, 3), padding='same')(x)
            x = layers.Flatten()(x)
            x = layers.Dropout(0.3)(x)
            # x = layers.Dense(128, activation='relu')(x)
            # x = layers.Dense(256, activation='linear')(x)
            # x = layers.Dropout(0.2)(x)
            output_val = layers.Dense(num_class, activation='softmax')(x)
        else:
            x = layers.Conv1D(128, 50, strides=30, padding='same')(x)
            # x = layers.Conv1D(256, 5, strides=3, padding='same')(x)
            # x = self.pooling_layer(x)
            # x = self.pooling_layer(x)
            output_val = x

        return output_val


#%% __main__
if __name__ == "__main__":
    print("hello, world~!")
