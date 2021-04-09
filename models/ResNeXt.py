
import tensorflow as tf
from tensorflow.keras import layers


##
class resnext_layers_a(layers.Layer):
    def __init__(self, **kwarg):
        super(resnext_layers_a, self).__init__()

        if "channel_size" in kwarg.keys():
            self.chan_size = kwarg['channel_size']

    def __call__(self, inputs, **kwarg):
        conv2d_layer_1 = layers.Conv2D(4, (1, 1), padding='same')
        conv2d_layer_2 = layers.Conv2D(4, (3, 3), padding='same')
        conv2d_layer_3 = layers.Conv2D(self.chan_size, (1, 1), padding='same')

        init_val = inputs
        # inputs = layers.BatchNormalization()(inputs)

        x = conv2d_layer_1(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = conv2d_layer_2(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = conv2d_layer_3(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        # x = tf.nn.relu(x)
        # x = layers.MaxPooling2D(pool_size=(2, 1), padding='same')(x)
        # x = layers.Dropout(0.2)(x)
        x = layers.Dropout(0.2)(x)

        return x


##
class resnext_layers_b(layers.Layer):
    def __init__(self, **kwarg):
        super(resnext_layers_b, self).__init__()

    def __call__(self, inputs, **kwarg):
        conv2d_layer_1 = layers.Conv2D(4, (1, 1), padding='same')
        conv2d_layer_2 = layers.Conv2D(4, (3, 3), padding='same')

        x = conv2d_layer_1(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = conv2d_layer_2(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = layers.Dropout(0.2)(x)

        return x


##
class resnext_layers_c(layers.Layer):
    def __init__(self, **kwarg):
        super(resnext_layers_c, self).__init__()

    def __call__(self, inputs, **kwarg):
        conv2d_layer_1 = layers.Conv2D(128, (1, 1), padding='same')
        conv2d_layer_3 = layers.Conv2D(256, (1, 1), padding='same')

        x = conv2d_layer_1(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25,s26,s27,s28,s29,s30,s31 = tf.split(x, num_or_size_splits=32, axis=-1)

        x0 = layers.Conv2D(4, (3, 3), padding='same')(s0)
        x1 = layers.Conv2D(4, (3, 3), padding='same')(s1)
        x2 = layers.Conv2D(4, (3, 3), padding='same')(s2)
        x3 = layers.Conv2D(4, (3, 3), padding='same')(s3)
        x4 = layers.Conv2D(4, (3, 3), padding='same')(s4)
        x5 = layers.Conv2D(4, (3, 3), padding='same')(s5)
        x6 = layers.Conv2D(4, (3, 3), padding='same')(s6)
        x7 = layers.Conv2D(4, (3, 3), padding='same')(s7)
        x8 = layers.Conv2D(4, (3, 3), padding='same')(s8)
        x9 = layers.Conv2D(4, (3, 3), padding='same')(s9)
        x10 = layers.Conv2D(4, (3, 3), padding='same')(s10)
        x11 = layers.Conv2D(4, (3, 3), padding='same')(s11)
        x12 = layers.Conv2D(4, (3, 3), padding='same')(s12)
        x13 = layers.Conv2D(4, (3, 3), padding='same')(s13)
        x14 = layers.Conv2D(4, (3, 3), padding='same')(s14)
        x15 = layers.Conv2D(4, (3, 3), padding='same')(s15)
        x16 = layers.Conv2D(4, (3, 3), padding='same')(s16)
        x17 = layers.Conv2D(4, (3, 3), padding='same')(s17)
        x18 = layers.Conv2D(4, (3, 3), padding='same')(s18)
        x19 = layers.Conv2D(4, (3, 3), padding='same')(s19)
        x20 = layers.Conv2D(4, (3, 3), padding='same')(s20)
        x21 = layers.Conv2D(4, (3, 3), padding='same')(s21)
        x22 = layers.Conv2D(4, (3, 3), padding='same')(s22)
        x23 = layers.Conv2D(4, (3, 3), padding='same')(s23)
        x24 = layers.Conv2D(4, (3, 3), padding='same')(s24)
        x25 = layers.Conv2D(4, (3, 3), padding='same')(s25)
        x26 = layers.Conv2D(4, (3, 3), padding='same')(s26)
        x27 = layers.Conv2D(4, (3, 3), padding='same')(s27)
        x28 = layers.Conv2D(4, (3, 3), padding='same')(s28)
        x29 = layers.Conv2D(4, (3, 3), padding='same')(s29)
        x30 = layers.Conv2D(4, (3, 3), padding='same')(s30)
        x31 = layers.Conv2D(4, (3, 3), padding='same')(s31)

        output_list = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31]

        x = tf.concat(output_list, -1)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_3(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = layers.Dropout(0.2)(x)

        return x


##
class resnext_block_a(layers.Layer):
    def __init__(self, **kwarg):
        super(resnext_block_a, self).__init__()

    def __call__(self, inputs, **kwarg):
        one_path_0 = resnext_layers_a(channel_size = 256)
        one_path_1 = resnext_layers_a(channel_size = 256)
        one_path_2 = resnext_layers_a(channel_size = 256)
        one_path_3 = resnext_layers_a(channel_size = 256)
        one_path_4 = resnext_layers_a(channel_size = 256)
        one_path_5 = resnext_layers_a(channel_size = 256)
        one_path_6 = resnext_layers_a(channel_size = 256)
        one_path_7 = resnext_layers_a(channel_size = 256)
        one_path_8 = resnext_layers_a(channel_size = 256)
        one_path_9 = resnext_layers_a(channel_size = 256)
        one_path_10 = resnext_layers_a(channel_size = 256)
        one_path_11 = resnext_layers_a(channel_size = 256)
        one_path_12 = resnext_layers_a(channel_size = 256)
        one_path_13 = resnext_layers_a(channel_size = 256)
        one_path_14 = resnext_layers_a(channel_size = 256)
        one_path_15 = resnext_layers_a(channel_size = 256)
        one_path_16 = resnext_layers_a(channel_size = 256)
        one_path_17 = resnext_layers_a(channel_size = 256)
        one_path_18 = resnext_layers_a(channel_size = 256)
        one_path_19 = resnext_layers_a(channel_size = 256)
        one_path_20 = resnext_layers_a(channel_size = 256)
        one_path_21 = resnext_layers_a(channel_size = 256)
        one_path_22 = resnext_layers_a(channel_size = 256)
        one_path_23 = resnext_layers_a(channel_size = 256)
        one_path_24 = resnext_layers_a(channel_size = 256)
        one_path_25 = resnext_layers_a(channel_size = 256)
        one_path_26 = resnext_layers_a(channel_size = 256)
        one_path_27 = resnext_layers_a(channel_size = 256)
        one_path_28 = resnext_layers_a(channel_size = 256)
        one_path_29 = resnext_layers_a(channel_size = 256)
        one_path_30 = resnext_layers_a(channel_size = 256)
        one_path_31 = resnext_layers_a(channel_size = 256)

        x_0 = tf.identiry(inputs)

        y_0 = one_path_0(x_0)
        y_1 = one_path_1(x_0)
        y_2 = one_path_2(x_0)
        y_3 = one_path_3(x_0)
        y_4 = one_path_4(x_0)
        y_5 = one_path_5(x_0)
        y_6 = one_path_6(x_0)
        y_7 = one_path_7(x_0)
        y_8 = one_path_8(x_0)
        y_9 = one_path_9(x_0)
        y_10 = one_path_10(x_0)
        y_11 = one_path_11(x_0)
        y_12 = one_path_12(x_0)
        y_13 = one_path_13(x_0)
        y_14 = one_path_14(x_0)
        y_15 = one_path_15(x_0)
        y_16 = one_path_16(x_0)
        y_17 = one_path_17(x_0)
        y_18 = one_path_18(x_0)
        y_19 = one_path_19(x_0)
        y_20 = one_path_20(x_0)
        y_21 = one_path_21(x_0)
        y_22 = one_path_22(x_0)
        y_23 = one_path_23(x_0)
        y_24 = one_path_24(x_0)
        y_25 = one_path_25(x_0)
        y_26 = one_path_26(x_0)
        y_27 = one_path_27(x_0)
        y_28 = one_path_28(x_0)
        y_29 = one_path_29(x_0)
        y_30 = one_path_30(x_0)
        y_31 = one_path_31(x_0)

        result = tf.math.add_n([y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11, y_12, y_13, y_14, y_15, y_16, y_17, y_18, y_19, y_20, y_21, y_22, y_23, y_24, y_25, y_26, y_27, y_28, y_29, y_30, y_31])

        x_1 = tf.math.add(result, inputs)
        x_result = tf.nn.relu(x_1)

        return x_result


##
class resnext_block_b(layers.Layer):
    def __init__(self, **kwarg):
        super(resnext_block_b, self).__init__()

    def __call__(self, inputs, **kwarg):
        one_path_0 = resnext_layers_b()
        one_path_1 = resnext_layers_b()
        one_path_2 = resnext_layers_b()
        one_path_3 = resnext_layers_b()
        one_path_4 = resnext_layers_b()
        one_path_5 = resnext_layers_b()
        one_path_6 = resnext_layers_b()
        one_path_7 = resnext_layers_b()
        one_path_8 = resnext_layers_b()
        one_path_9 = resnext_layers_b()
        one_path_10 = resnext_layers_b()
        one_path_11 = resnext_layers_b()
        one_path_12 = resnext_layers_b()
        one_path_13 = resnext_layers_b()
        one_path_14 = resnext_layers_b()
        one_path_15 = resnext_layers_b()
        one_path_16 = resnext_layers_b()
        one_path_17 = resnext_layers_b()
        one_path_18 = resnext_layers_b()
        one_path_19 = resnext_layers_b()
        one_path_20 = resnext_layers_b()
        one_path_21 = resnext_layers_b()
        one_path_22 = resnext_layers_b()
        one_path_23 = resnext_layers_b()
        one_path_24 = resnext_layers_b()
        one_path_25 = resnext_layers_b()
        one_path_26 = resnext_layers_b()
        one_path_27 = resnext_layers_b()
        one_path_28 = resnext_layers_b()
        one_path_29 = resnext_layers_b()
        one_path_30 = resnext_layers_b()
        one_path_31 = resnext_layers_b()

        x_0 = tf.identiry(inputs)

        y_0 = one_path_0(x_0)
        y_1 = one_path_1(x_0)
        y_2 = one_path_2(x_0)
        y_3 = one_path_3(x_0)
        y_4 = one_path_4(x_0)
        y_5 = one_path_5(x_0)
        y_6 = one_path_6(x_0)
        y_7 = one_path_7(x_0)
        y_8 = one_path_8(x_0)
        y_9 = one_path_9(x_0)
        y_10 = one_path_10(x_0)
        y_11 = one_path_11(x_0)
        y_12 = one_path_12(x_0)
        y_13 = one_path_13(x_0)
        y_14 = one_path_14(x_0)
        y_15 = one_path_15(x_0)
        y_16 = one_path_16(x_0)
        y_17 = one_path_17(x_0)
        y_18 = one_path_18(x_0)
        y_19 = one_path_19(x_0)
        y_20 = one_path_20(x_0)
        y_21 = one_path_21(x_0)
        y_22 = one_path_22(x_0)
        y_23 = one_path_23(x_0)
        y_24 = one_path_24(x_0)
        y_25 = one_path_25(x_0)
        y_26 = one_path_26(x_0)
        y_27 = one_path_27(x_0)
        y_28 = one_path_28(x_0)
        y_29 = one_path_29(x_0)
        y_30 = one_path_30(x_0)
        y_31 = one_path_31(x_0)

        result = tf.concat([y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11, y_12, y_13, y_14, y_15, y_16, y_17, y_18, y_19, y_20, y_21, y_22, y_23, y_24, y_25, y_26, y_27, y_28, y_29, y_30, y_31])

        x = layers.Conv2D(256, (1, 1), padding='same')(result)

        x_1 = tf.math.add(x, inputs)
        x_result = tf.nn.relu(x_1)

        return x_result



##
class resnext_block_c(layers.Layer):
    def __init__(self, **kwarg):
        super(resnext_block_c, self).__init__()

    def __call__(self, inputs, **kwarg):

        conv2d_layer_1 = layers.Conv2D(128, (1, 1), padding='same')
        conv2d_layer_2 = resnext_layers_c()
        conv2d_layer_3 = layers.Conv2D(256, (1, 1), padding='same')

        x_0 = tf.identiry(inputs)

        x = conv2d_layer_1(x_0)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_2(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_3(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        result = tf.math.add(x, inputs)
        result = tf.nn.relu(result)

        return result


##
if __name__ == '__main__':
    print("hello, world~!!")



## endl
