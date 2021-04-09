# -*- coding: utf-8 -*-

import sys
sys.path.append("C:\\Users\\jyback_pnc\\Desktop\\code\\girl_from_ipanema")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from draw_graph import result_graph as rg
from module import save_the_best as save_best
from make_train_data import load_train_data_class as load_data_cl

from variables import raw_list, mfcc_list_13, mfcc_list_26, logfb_list
from variables import normal_mfcc13_list, normal_mfcc26_list, normal_logfb_list, normal_raw_sig

train_data_path = "../"

file_list = normal_raw_sig

import basic_rnn_class as brnncl
import basic_cnn_block as bcblock
import residual_block as resnet_block
import residual_block_1D as resnet_1D

import tensorflow as tf
from tensorflow.keras import layers

#%%
class dense_class(layers.Layer):
    def __init__(self):
        super(dense_class, self).__init__()
        
        self.dense_1 = layers.Dense(128, activation='relu')
        self.dense_2 = layers.Dense(64, activation='relu')
        self.softmax = layers.Dense(10, activation='softmax')
        self.flat = layers.Flatten()
        
    def call(self, x):
        x = self.flat(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.softmax(x)
        
        return x
    
#%%
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# train_images = tf.expand_dims(train_images, -1)
# print("data shape : "+ str(train_images.shape))
# test_images = tf.expand_dims(test_images, -1)
# print("data shape : "+ str(test_images.shape))

train_images = tf.cast(train_images, tf.float32) / 255.0

test_images = tf.cast(test_images, tf.float32) / 255.0

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])


#%%

# input_vec = tf.keras.Input(shape=[28, 28, 1])
input_vec = tf.keras.Input(shape=[28, 28])

dense_layer = dense_class()

answer = dense_layer(input_vec)


# print(conv_shape)
# resnet = resnet_block.residual_net(pooling_bool=False, kernel_size=(3, 3))

# resnet = resnet_1D.residual_net_1D(pooling_bool=False, kernel_size=5, strides_size=1)

# answer = resnet(input_vec, num_of_classes=10, dense_softmax=True)

model = tf.keras.Model(inputs=input_vec, outputs=answer)

model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=20, batch_size=32)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\n테스트 정확도:', test_acc)