

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
import transformer

#%%
class cnn_block_1D_seq(layers.Layer):
    def __init__(self, **kwarg):
        super(cnn_block_1D_seq, self).__init__()
        if 'kernel_size' in kwarg.keys():
            k_n = kwarg['kernel_size']
        if 'strides_size' in kwarg.keys():
            s_n = kwarg['strides_size']

        self.k_num = k_n
        self.s_num = s_n

    def __call__(self, input_t, **kwarg):

        model = keras.Sequential(
            [
                layers.Conv1D(4, self.k_num, strides=self.s_num, padding='same'),
                layers.Conv1D(8, self.k_num, strides=self.s_num, padding='same'),
                layers.BatchNormalization(),
                layers.Conv1D(16, self.k_num, strides=self.s_num, padding='same'),
                layers.Conv1D(32, self.k_num, strides=self.s_num, padding='same'),

                # layers.AveragePooling1D(2, padding='same'),
                # layers.MaxPool1D(2, padding='same'),
                layers.BatchNormalization(),
                layers.Conv1D(64, self.k_num, strides=self.s_num, padding='same'),

                # layers.AveragePooling1D(2, padding='same'),
                # layers.MaxPool1D(2, padding='same'),
                # layers.LayerNormalization(),

                layers.Conv1D(96, self.k_num, strides=self.s_num, padding='same'),
                layers.BatchNormalization(),
                # layers.AveragePooling1D(2, padding='same'),
                # layers.MaxPool1D(2, padding='same'),

                layers.Conv1D(128, self.k_num, strides=self.s_num, padding='same'),

                # layers.AveragePooling1D(2, padding='same'),
                # layers.MaxPool1D(2, padding='same'),
                # layers.LayerNormalization(),
                layers.BatchNormalization(),

                layers.Conv1D(192, self.k_num, strides=self.s_num, padding='same'),
                layers.BatchNormalization(),
                # layers.AveragePooling1D(2, padding='same'),
                # layers.MaxPool1D(2, padding='same'),

                layers.Conv1D(256, self.k_num, strides=self.s_num, padding='same'),

                # layers.AveragePooling1D(2, padding='same'),
                # layers.MaxPool1D(2, padding='same'),

                # layers.LayerNormalization(),
                layers.BatchNormalization(),
                ]
            )

        x = model(input_t)
        return x

#%%
class cnn_block_1D(layers.Layer):
    def __init__(self, **kwarg):
        super(cnn_block_1D, self).__init__()
        if 'kernel_size' in kwarg.keys():
            k_n = kwarg['kernel_size']
        if 'strides_size' in kwarg.keys():
            s_n = kwarg['strides_size']

        self.k_num = k_n
        self.s_num = s_n

    def __call__(self, input_t, **kwarg):
        x = input_t

        x = layers.Conv1D(8, self.k_num, strides=self.s_num, padding='same')(x)

        # x = layers.AveragePooling1D(2, padding='same')(x)
        # x = layers.MaxPool1D(2, padding='same')(x)

        x = layers.Conv1D(16, self.k_num, strides=self.s_num, padding='same')(x)

        # x = layers.AveragePooling1D(2, padding='same')(x)
        # x = layers.MaxPool1D(2, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv1D(32, self.k_num, strides=self.s_num, padding='same')(x)

        # x = layers.AveragePooling1D(2, padding='same')(x)
        # x = layers.MaxPool1D(2, padding='same')(x)

        x = layers.Conv1D(64, self.k_num, strides=self.s_num, padding='same')(x)

        # x = layers.AveragePooling1D(2, padding='same')(x)
        # x = layers.MaxPool1D(2, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv1D(96, self.k_num, strides=self.s_num, padding='same')(x)

        # x = layers.AveragePooling1D(2, padding='same')(x)
        # x = layers.MaxPool1D(2, padding='same')(x)

        x = layers.Conv1D(128, self.k_num, strides=self.s_num, padding='same')(x)

        # x = layers.AveragePooling1D(2, padding='same')(x)
        # x = layers.MaxPool1D(2, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv1D(256, self.k_num, strides=self.s_num, padding='same')(x)

        return x


#%%
class cnn_block(layers.Layer):
    def __init__(self, **kwarg):
        super(cnn_block, self).__init__()
        if 'channel_size' in kwarg.keys():
            self.c_size = kwarg['channel_size']
        if 'kernel_size' in kwarg.keys():
            self.k_size = kwarg['kernel_size']
        self.rate = 0.2

        self.cnn_layer_1 = layers.Conv2D(self.c_size, self.k_size, padding='same')
        self.cnn_layer_2 = layers.Conv2D(self.c_size*2, self.k_size, padding='same')
        self.cnn_layer_3 = layers.Conv2D(self.c_size*2*2, self.k_size, padding='same')

        self.pooling_layer_1 = layers.MaxPool2D(pool_size=(2, 2), padding='same')
        self.pooling_layer_2 = layers.MaxPool2D(pool_size=(1, 2), padding='same')
        self.pooling_layer_3 = layers.MaxPool2D(pool_size=(4, 2), padding='same')
        self.pooling_layer_4 = layers.MaxPool2D(pool_size=(2, 1), padding='same')

        self.pooling_layer_5 = layers.MaxPool2D(pool_size=(2, 2), padding='same')
        self.pooling_layer_6 = layers.MaxPool2D(pool_size=(2, 2), padding='same')
        self.pooling_layer_7 = layers.MaxPool2D(pool_size=(2, 2), padding='same')


    def __call__(self, input_t, **kwarg):

        x = self.cnn_layer_1(input_t)
        x = self.pooling_layer_1(x)
        x = self.cnn_layer_2(x)
        x = self.pooling_layer_1(x)
        x = self.cnn_layer_3(x)
        x = self.pooling_layer_1(x)

        return x


#%%
class residual_net(layers.Layer):
    def __init__(self, **kwarg):
        super(residual_net, self).__init__()

        if 'channel_size' in kwarg.keys():
            self.c_size = kwarg['channel_size']
        if 'kernel_size' in kwarg.keys():
            self.k_size = kwarg['kernel_size']
        self.rate = 0.2

        self.cnn_layer_1 = layers.Conv2D(self.c_size, self.k_size, padding='same')
        self.cnn_layer_2 = layers.Conv2D(self.c_size, self.k_size, padding='same')

        self.pooling_layer_1 = layers.MaxPool2D(pool_size=(2, 2), padding='same')
        self.pooling_layer_2 = layers.MaxPool2D(pool_size=(1, 2), padding='same')
        self.pooling_layer_3 = layers.MaxPool2D(pool_size=(4, 2), padding='same')
        self.pooling_layer_4 = layers.MaxPool2D(pool_size=(2, 1), padding='same')
        self.pooling_layer_5 = layers.MaxPool2D(pool_size=(1, 4), padding='same')


    def __call__(self, input_t, **kwarg):

        x = self.cnn_layer_1(input_t)
        x = tf.nn.relu(x)
        x = self.cnn_layer_2(x)
        y = layers.Conv2D(self.c_size, (1, 1), padding='same')(input_t)
        x = tf.math.add(y, x)
        x = tf.nn.relu(x)
        x = layers.Dropout(self.rate)(x)

        return x


#%% class -> residual cnn block
class Multihead_Attention_Block_1(layers.Layer):
    def __init__(self, **kwarg):
        super(Multihead_Attention_Block_1, self).__init__()

    def __call__(self, input_t):
        mha_layer_1 = transformer.EncoderLayer_no_ffn(input_t.shape[2], 8, 512)
        x = mha_layer_1(input_t, False, None)
        # x = layers.BatchNormalization()(x)
        # x = tf.nn.relu(x)

        # x = self.expanding_dims(x)
        return x


#%% class -> residual cnn block
class Multihead_Attention_Block_2(layers.Layer):
    def __init__(self, **kwarg):
        super(Multihead_Attention_Block_2, self).__init__()

    def __call__(self, input_t):
        # x = self.convert_shape(input_t)
        mha_layer_1 = transformer.EncoderLayer_with_ffn(input_t.shape[2], 8, 512)
        x = mha_layer_1(input_t, False, None)
        # x = layers.BatchNormalization()(x)
        # x = tf.nn.relu(x)

        return x


#%%
class Multihead_Attention_layer(layers.Layer):
    def __init__(self, **kwarg):
        super(Multihead_Attention_layer, self).__init__()

    def convert_shape(self, input_t):
        print(tf.shape(input_t))
        a = input_t.shape
        print("convert_shape function")
        output_t = tf.reshape(input_t, [-1, a[1], a[2]*a[3]])
        print(output_t.shape)
        return output_t

    def expanding_dims(self, input_t):
        x = tf.expand_dims(input_t, -1)
        print(x.shape)
        return x


    def __call__(self, input_t, **kwarg):

        resnet_layer_1 = residual_net(channel_size=32, kernel_size = (3, 3))
        resnet_layer_2 = residual_net(channel_size=64, kernel_size = (3, 3))
        resnet_layer_3 = residual_net(channel_size=128, kernel_size = (3, 3))
        resnet_layer_4 = residual_net(channel_size=256, kernel_size = (3, 3))

        resnet_layer_5 = residual_net(channel_size=32, kernel_size = (3, 3))
        resnet_layer_6 = residual_net(channel_size=64, kernel_size = (3, 3))
        resnet_layer_7 = residual_net(channel_size=128, kernel_size = (3, 3))
        resnet_layer_8 = residual_net(channel_size=256, kernel_size = (3, 3))

        resnet_layer_9 = residual_net(channel_size=512, kernel_size = (3, 3))

        mhab_layer_1 = Multihead_Attention_Block_1()
        mhab_layer_2 = Multihead_Attention_Block_1()
        mhab_layer_3 = Multihead_Attention_Block_1()

        mhab_layer_4 = Multihead_Attention_Block_2()
        mhab_layer_5 = Multihead_Attention_Block_2()
        mhab_layer_6 = Multihead_Attention_Block_2()
        mhab_layer_7 = Multihead_Attention_Block_2()

        cnn_1D_layer = cnn_block_1D_seq(kernel_size=5, strides_size=2)
        cnn_1D_layer_2 = cnn_block_1D(kernel_size=5, strides_size=2)

        x = input_t

        # x = cnn_1D_layer(x)
        # x = cnn_1D_layer_2(x)
        print(x.shape)

        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        # x = resnet_layer_1(x)
        x = layers.MaxPool2D((2, 2), padding='same')(x)
        # x = tf.nn.relu(x)
        x = layers.BatchNormalization()(x)
        # x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        # x = resnet_layer_2(x)
        x = layers.MaxPool2D((2, 2), padding='same')(x)
        # x = tf.nn.relu(x)
        # x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        # x = resnet_layer_3(x)
        x = layers.MaxPool2D((2, 2), padding='same')(x)
        # x = layers.MaxPool2D((4, 4), padding='same')(x)
        # x = tf.nn.relu(x)
        x = layers.BatchNormalization()(x)
        # x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        # x = resnet_layer_4(x)
        x = layers.MaxPool2D((2, 2), padding='same')(x)
        # x = layers.MaxPool2D((4, 4), padding='same')(x)
        # x = tf.nn.relu(x)
        # x = layers.Dropout(0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        # x = layers.LayerNormalization()(x)

        y = self.convert_shape(x)

        # y = mhab_layer_1(y)
        # y = mhab_layer_2(y)
        # y = mhab_layer_3(y)
        y = mhab_layer_4(y)
        y = mhab_layer_5(y)
        y = mhab_layer_6(y)
        # y = mhab_layer_7(y)

        # z = layers.Conv2D(256, (3, 3), padding='same')(x)
        # z = tf.nn.relu(z)
        # z = layers.BatchNormalization()(z)
        # z = layers.Dropout(0.2)(z)

        # z = layers.Conv2D(256, (3, 3), padding='same')(z)
        # z = tf.nn.relu(z)
        # z = layers.Dropout(0.2)(z)
        # z = layers.BatchNormalization()(z)
        # z = layers.Dropout(0.2)(z)

        z = resnet_layer_8(x)
        # z = layers.BatchNormalization()(z)
        # z = resnet_layer_4(z)
        # z = layers.LayerNormalization()(z)

        # z = layers.Dropout(0.2)(z)

        z = self.convert_shape(z)

        x = tf.math.add(y, z)
        # w = tf.math.add(y, z)
        # x = tf.math.add(v, w)
        x = tf.nn.relu(x)

        x = layers.BatchNormalization()(x)
        # x = layers.LayerNormalization()(x)

        # x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LSTM(256)(x)
        # x = layers.Bidirectional(layers.LSTM(128))(x)

        # x = layers.GlobalAveragePooling2D()(x)
        x = layers.Flatten()(x)
        # x = layers.Dropout(0.2)(x)
        # x = layers.Dense(256, activation='linear')(x)
        x = layers.Dropout(0.3)(x)
        # x = layers.Dense(128, activation='relu')(x)
        # x = layers.Dropout(0.2)(x)
        # x = layers.BatchNormalization()(x)

        return x


## only for cnn type input
max_number = 200
def load_train_data(*args, **kwarg):

    train_load_data = np.load('D:\\mod_train_data.npz', allow_pickle=True)
    test_load_data = np.load('D:\\mod_test_data.npz', allow_pickle=True)

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
epoch_num = 100
num_label = 17

load_mode = 0


#%% loading data
train_mfcc_feats, test_mfcc_feats, train_labels, test_labels,conv_shape=load_train_data()


#%% build model
input_vec = tf.keras.Input(shape=conv_shape)

mha_layer = Multihead_Attention_layer()
x = mha_layer(input_vec)

answer = layers.Dense(num_label, activation='softmax')(x)

model = tf.keras.Model(inputs=input_vec, outputs=answer)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#%% epoch training loop
max_acc = 0.0
graph_cl = rg()

#%% old train // test per one epoch
for i,epo in enumerate(range(epoch_num)):
    print("\n\n%d th epoch\n"%(epo+1))

    if load_mode == 1:
        model.load_weights('mha_resnet.h5')

    history = model.fit(train_mfcc_feats, train_labels,
                            batch_size=num_batch, epochs=1, verbose=1)

    loss, metric_res = model.evaluate(x=test_mfcc_feats,
            y=test_labels, verbose=1)   # "return_dict" argument doesn't work in tf 2.1
                                        # but, after 2.2, it workss

    model.save('mha_resnet.h5')
    if metric_res > max_acc:
        max_acc = metric_res
        model.save('mha_resnet_best.h5')
        print("recent max acc value : ", max_acc)

    graph_cl.make_list(train_result=history, eval_loss=loss, eval_acc=metric_res)


graph_cl.draw_plt_graph()
print("\nhighest accuracy in this model only : ", max_acc, '\n\n\n')











## endl
