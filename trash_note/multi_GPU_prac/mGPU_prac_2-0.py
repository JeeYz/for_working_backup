import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from scipy import fftpack
from scipy import signal

from tensorflow.keras.layers.experimental import preprocessing

import librosa

train_data_file = "D:\\train_data_.npz"

loaded_data = np.load(train_data_file)


train_data_1 = loaded_data['data'][:10]

print(type(train_data_1))


def stft_func(input_sig, **kwargs):

    train_data_np = list()

    frame_size = 255
    delay_size = 128

    gap_sizes = frame_size - delay_size
    range_len = len(input_sig)//(frame_size-delay_size)

    for num in range(range_len):

        one_data = input_sig[num*gap_sizes:(num+1)*gap_sizes]

        res = np.fft.fft(one_data)
        x = np.abs(res)

        train_data_np.append(x)
        
    return np.array(train_data_np)


train_data_x = list()

for i, one_data in enumerate(train_data_1):

    result = stft_func(one_data)
    train_data_x.append(result)

    print("\r{}th file is done...".format(i+1), end='')

train_label = np.array(train_data_x)
train_label = np.expand_dims(train_label, axis=-1)
train_label = preprocessing.Resizing(32, 32)(train_label)
# print(train_label)

train_label = train_label.numpy()

# train_data_1 = tf.data.Dataset.from_tensor_slices(train_data_1)
# train_label = tf.data.Dataset.from_tensor_slices(train_label)


train_data = tf.data.Dataset.from_tensor_slices((train_data_1, train_label)).shuffle(5000).batch(4)

# print(train_data)

import time



input_sig = keras.Input(shape=(64000,))

# x = tf.expand_dims(input_sig, -1)
# x = tf.broadcast_to(x, [4, 64000, 128])
# x = tf.expand_dims(x, -1)

x = tf.expand_dims(input_sig, -1)
x = tf.expand_dims(x, -1)

x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((4, 1))(x)

x = tf.reshape(x, (-1, x.shape[1], x.shape[3]))
x = tf.expand_dims(x, -1)

x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((4, 1))(x)

x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((4, 1))(x)

x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((4, 1))(x)

x = layers.BatchNormalization()(x)
x = layers.Conv2D(1, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((4, 1))(x)

answer = preprocessing.Resizing(32, 32)(x)

# print(answer)

model = keras.Model(inputs=input_sig, outputs=answer)

model.summary()

keras.utils.plot_model(model, "proto_model_with_shape_info.png", show_shapes=True)

model.compile(
    # loss=keras.losses.BinaryCrossentropy(),
    loss='mse',
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

history = model.fit(train_data, epochs=10)

# history = model.fit(train_data_1, train_label, epochs=10)


model.save("D:\\model_mGPU_prac_2-0.h5")
