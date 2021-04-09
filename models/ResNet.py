
import tensorflow as tf
from tensorflow.keras import layers

'''
ResNet Block for copy
'''


##
class residual_net_block_normal(layers.Layer):

    def __init__(self, **kwarg):
        super(residual_net_block_normal, self).__init__()
        if "channel_size" in kwarg.keys():
            self.chan_size = kwarg['channel_size']

    def __call__(self, inputs, **kwarg):
        conv2d_layer_1 = layers.Conv2D(self.chan_size[0], (3, 3),
                                        padding='same')
        conv2d_layer_2 = layers.Conv2D(self.chan_size[1], (3, 3),
                                        padding='same')

        # init_val = inputs
        init_val = tf.identity(inputs)

        # inputs = layers.BatchNormalization()(inputs)

        x = conv2d_layer_1(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_2(x)
        x = layers.BatchNormalization()(x)

        y = layers.Conv2D(self.chan_size[1], (1, 1), padding='same')(init_val)
        x = tf.math.add(y, x)
        x = tf.nn.relu(x)
        # x = layers.MaxPooling2D(pool_size=(2, 1), padding='same')(x)
        x = layers.Dropout(0.2)(x)

        return x


##
class residual_net_block_bottle(layers.Layer):

    def __init__(self, **kwarg):
        super(residual_net_block_bottle, self).__init__()
        if "filter_size" in kwarg.keys():
            self.fil_size = kwarg['filter_size']
        if "channel_size" in kwarg.keys():
            self.ch_size = kwarg['channel_size']

    def __call__(self, inputs, **kwarg):
        conv2d_layer_1 = layers.Conv2D(self.ch_size[0], (1, 1),
                                        padding='same')
        conv2d_layer_2 = layers.Conv2D(self.ch_size[0], (3, 3),
                                        padding='same')
        conv2d_layer_3 = layers.Conv2D(self.ch_size[1], (1, 1),
                                        padding='same')

        # init_val = inputs
        init_val = tf.identity(inputs)
        # inputs = layers.BatchNormalization()(inputs)

        x = conv2d_layer_1(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_2(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = conv2d_layer_3(x)
        x = layers.BatchNormalization()(x)

        y = layers.Conv2D(self.chan_size[1], (1, 1), padding='same')(init_val)
        x = tf.math.add(y, x)
        x = tf.nn.relu(x)
        # x = layers.MaxPooling2D(pool_size=(2, 1), padding='same')(x)
        x = layers.Dropout(0.2)(x)

        return x





##
if __name__ == '__main__':
    print("hello, world~!!")



## endl
