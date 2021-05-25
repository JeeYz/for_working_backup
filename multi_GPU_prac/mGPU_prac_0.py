import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python import framework
from tensorflow.python.framework.tensor_util import constant_value


# @tf.function
def stft_func(input_sig, **kwargs):

    if "frame_size" in kwargs.keys():
        frame_size = kwargs['frame_size']
    if "delay_size" in kwargs.keys():
        delay_size = kwargs['delay_size']

    gap_sizes = frame_size - delay_size
    range_len = len(input_sig)//(frame_size-delay_size)
    result = list()

    for num in range(range_len+1):

        one_data = input_sig[num*gap_sizes:(num+1)*gap_sizes]

        if num == 0:
            max_num = len(one_data)

        # one_data = input_sig[num*gap_sizes:(num+1)*gap_sizes]

        if max_num > len(one_data):
            one_data = np.pad(one_data, (0, (max_num-len(one_data))))
            # print(one_data)

        x = tf.cast(one_data, tf.complex64)
        tres = tf.signal.fft(x)
        
        x = tf.abs(tres)
        x = tf.cast(x, tf.float32)
        result.append(x)
    return np.array(result)

train_data = list()
train_data_size = 1000

for i in range(train_data_size):
    temp_data = np.random.randint(-3000, 3000, size=32000)
    result = stft_func(temp_data, frame_size = 255, delay_size = 128)
    train_data.append(result)

train_data = np.array(train_data)

result = np.expand_dims(train_data, -1)
print(result.shape)

input_data = tf.data.Dataset.from_tensor_slices(result)
print(input_data)

input_sig = keras.Input(shape=(252, 127, 1))

x = preprocessing.Resizing(32, 32)(input_sig)
print(x.shape)
x = preprocessing.Normalization()(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
answer = layers.Dense(6, activation='softmax')(x)



