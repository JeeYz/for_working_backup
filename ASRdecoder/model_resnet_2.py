
import sys

temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from draw_graph import result_graph as rg

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

        self.residual_cnn_layer_1_0 = residual_cnn_block_2D(channel_size=[8, 8])
        self.residual_cnn_layer_2_0 = residual_cnn_block_2D(channel_size=[16, 16])
        self.residual_cnn_layer_2_1 = residual_cnn_block_2D(channel_size=[16, 16])
        self.residual_cnn_layer_2_2 = residual_cnn_block_2D(channel_size=[16, 16])
        self.residual_cnn_layer_2_3 = residual_cnn_block_2D(channel_size=[16, 16])

        self.residual_cnn_layer_3_0 = residual_cnn_block_2D(channel_size=[32, 32])
        self.residual_cnn_layer_3_1 = residual_cnn_block_2D(channel_size=[32, 32])
        self.residual_cnn_layer_3_2 = residual_cnn_block_2D(channel_size=[32, 32])
        self.residual_cnn_layer_3_3 = residual_cnn_block_2D(channel_size=[32, 32])
        self.residual_cnn_layer_3_4 = residual_cnn_block_2D(channel_size=[32, 32])
        self.residual_cnn_layer_3_5 = residual_cnn_block_2D(channel_size=[32, 32])
        self.residual_cnn_layer_3_6 = residual_cnn_block_2D(channel_size=[32, 32])
        self.residual_cnn_layer_3_7 = residual_cnn_block_2D(channel_size=[32, 32])

        self.residual_cnn_layer_4_0 = residual_cnn_block_2D(channel_size=[64, 64])
        self.residual_cnn_layer_4_1 = residual_cnn_block_2D(channel_size=[64, 64])
        self.residual_cnn_layer_4_2 = residual_cnn_block_2D(channel_size=[64, 64])
        self.residual_cnn_layer_4_3 = residual_cnn_block_2D(channel_size=[64, 64])
        self.residual_cnn_layer_4_4 = residual_cnn_block_2D(channel_size=[64, 64])
        self.residual_cnn_layer_4_5 = residual_cnn_block_2D(channel_size=[64, 64])
        self.residual_cnn_layer_4_6 = residual_cnn_block_2D(channel_size=[64, 64])
        self.residual_cnn_layer_4_7 = residual_cnn_block_2D(channel_size=[64, 64])

        self.residual_cnn_layer_5_0 = residual_cnn_block_2D(channel_size=[128, 128])
        self.residual_cnn_layer_5_1 = residual_cnn_block_2D(channel_size=[128, 128])
        self.residual_cnn_layer_5_2 = residual_cnn_block_2D(channel_size=[128, 128])
        self.residual_cnn_layer_5_3 = residual_cnn_block_2D(channel_size=[128, 128])
        self.residual_cnn_layer_5_4 = residual_cnn_block_2D(channel_size=[128, 128])
        self.residual_cnn_layer_5_5 = residual_cnn_block_2D(channel_size=[128, 128])
        self.residual_cnn_layer_5_6 = residual_cnn_block_2D(channel_size=[128, 128])
        self.residual_cnn_layer_5_7 = residual_cnn_block_2D(channel_size=[128, 128])

        self.residual_cnn_layer_6_0 = residual_cnn_block_2D(channel_size=[256, 256])
        self.residual_cnn_layer_6_1 = residual_cnn_block_2D(channel_size=[256, 256])
        self.residual_cnn_layer_6_2 = residual_cnn_block_2D(channel_size=[256, 256])
        self.residual_cnn_layer_6_3 = residual_cnn_block_2D(channel_size=[256, 256])
        self.residual_cnn_layer_6_4 = residual_cnn_block_2D(channel_size=[256, 256])
        self.residual_cnn_layer_6_5 = residual_cnn_block_2D(channel_size=[256, 256])
        self.residual_cnn_layer_6_6 = residual_cnn_block_2D(channel_size=[256, 256])
        self.residual_cnn_layer_6_7 = residual_cnn_block_2D(channel_size=[256, 256])

        self.residual_cnn_layer_7_0 = residual_cnn_block_2D(channel_size=[512, 512])

        # self.pooling_layer = layers.MaxPool1D(pool_size=4, padding='same')
        self.pooling_layer = layers.MaxPooling2D(pool_size=(3, 1), padding='same')


    def __call__(self, inputs, **kwarg):

        if 'num_of_classes' in kwarg.keys():
            num_class = kwarg['num_of_classes']

        pooling_size_0 = (1, 2)
        pooling_size_1 = (2, 1)
        pooling_size_2 = (2, 2)

        x = self.residual_cnn_layer_3_0(inputs)
        # x = self.residual_cnn_layer_3_1(x)
        x = layers.MaxPooling2D(pool_size=pooling_size_2, padding='same')(x)

        x = self.residual_cnn_layer_4_0(x)
        # x = self.residual_cnn_layer_4_1(x)
        x = layers.MaxPooling2D(pool_size=pooling_size_2, padding='same')(x)

        x = self.residual_cnn_layer_5_0(x)
        # x = self.residual_cnn_layer_5_1(x)
        x = layers.MaxPooling2D(pool_size=pooling_size_2, padding='same')(x)

        x = self.residual_cnn_layer_5_4(x)
        # x = self.residual_cnn_layer_5_5(x)
        x = layers.MaxPooling2D(pool_size=pooling_size_2, padding='same')(x)

        x = self.residual_cnn_layer_6_0(x)
        # x = self.residual_cnn_layer_6_1(x)
        x = layers.MaxPooling2D(pool_size=pooling_size_2, padding='same')(x)

        x = self.residual_cnn_layer_7_0(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Flatten()(x)
        # x = layers.Dense(256)(x)
        x = layers.Dropout(0.3)(x)
        output_val = layers.Dense(num_class, activation='softmax')(x)

        return output_val


## only for cnn type input
max_number = 200

# train_data_path = 'D:\\mod_train_data.npz'
# test_data_path = 'D:\\mod_test_data.npz'

# train_data_path = 'D:\\aug_train_data.npz'
test_data_path = 'D:\\aug_test_data.npz'

# train_data_path = 'D:\\aug_norm_train_data.npz'
# test_data_path = 'D:\\aug_norm_test_data.npz'

train_data_path = 'D:\\train_data_for_all_with_zeroth.npz'

def load_train_data(*args, **kwarg):

    train_load_data = np.load(train_data_path, allow_pickle=True)
    test_load_data = np.load(test_data_path, allow_pickle=True)

    train_labels = train_load_data['label']
    train_feats = train_load_data['data']
    train_rates = train_load_data['rate']

    test_labels = test_load_data['label']
    test_feats = test_load_data['data']
    test_rates = test_load_data['rate']

    train_feats = tf.keras.preprocessing.sequence.pad_sequences(train_feats,
                                            maxlen=max_number, padding='post', dtype='float32')
    test_feats = tf.keras.preprocessing.sequence.pad_sequences(test_feats,
                                            maxlen=max_number, padding='post', dtype='float32')

    train_feats = tf.expand_dims(train_feats, -1)
    print("data shape : "+ str(train_feats.shape))
    test_feats = tf.expand_dims(test_feats, -1)
    print("data shape : "+ str(test_feats.shape))

    conv_shape = (train_feats.shape[1], train_feats.shape[2], 1)

    return train_feats, test_feats, train_labels, test_labels, conv_shape


num_batch = 64
epoch_num = 30
num_label = 17

load_mode = 0


#%% loading data
train_mfcc_feats, test_mfcc_feats, train_labels, test_labels,conv_shape=load_train_data()


#%% build model
input_vec = tf.keras.Input(shape=conv_shape)

resnet = residual_net_2D()

answer = resnet(input_vec, num_of_classes=num_label)

model = tf.keras.Model(inputs=input_vec, outputs=answer)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#%% epoch training loop
max_acc = 0.0
graph_cl = rg()

if load_mode == 1:
    model.load_weights('resnet_model.h5')

history = model.fit(train_mfcc_feats, train_labels,
                        batch_size=num_batch, epochs=epoch_num, verbose=1,
                        validation_split=0.1, shuffle=True)

loss, metric_res = model.evaluate(x=test_mfcc_feats,
        y=test_labels, verbose=1)   # "return_dict" argument doesn't work in tf 2.1
                                    # but, after 2.2, it workss

model.save('resnet_model.h5')

graph_cl.make_list(train_result=history, eval_loss=loss, eval_acc=metric_res)


graph_cl.draw_plt_graph()












## endl
