
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from scipy.io import wavfile

train_data_small_path = 'D:\\train_data_for_STFT_small.npz'

train_data_np = np.load(train_data_small_path, allow_pickle=True)

train_data = train_data_np['data']

new_label_list = list()
for one_train in train_data:
    result = tf.signal.stft(one_train, 
                            frame_length=255, 
                            frame_step=128
                            )
    # print(result.shape) -> (249, 129)
    new_label_list.append(result)

train_label = np.asarray(new_label_list)
# train_label = train_data_np['label']


# train_label = np.expand_dims(train_label, axis=-1)
print("\n******************************\n")
print(train_data.shape)
print(train_label.shape)
# print(train_label)
print("\n******************************\n")


input_vec = tf.keras.Input(shape=(32000,))

x = tf.abs(input_vec)
x = tf.expand_dims(x, -1)

x = layers.Conv1D(32, 3, strides=2, activation='relu')(x)
x = layers.Conv1D(32, 3, strides=1, 
                   padding='same', activation='relu')(x)
x = layers.MaxPool1D(2)(x)
# x = layers.LayerNormalization()(x)
x = layers.BatchNormalization()(x)
x = layers.Conv1D(64, 3, strides=2, activation='relu')(x)
x = layers.Conv1D(64, 3, strides=1, 
                    padding='same', activation='relu')(x)
x = layers.MaxPool1D(2)(x)
x = layers.Conv1D(64, 3, strides=2, activation='relu')(x)
x = layers.Conv1D(64, 3, strides=1, 
                    padding='same', activation='relu')(x)
x = layers.MaxPool1D(2)(x)
x = layers.BatchNormalization()(x)
# x = layers.LayerNormalization()(x)
# x = layers.Conv1D(129, 3, strides=2, activation='sigmoid')(x)
# answer = tf.expand_dims(x, axis=-1)
answer = layers.Conv1D(129, 3, strides=2, activation='relu')(x)


model = tf.keras.Model(inputs=input_vec, outputs=answer)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    # optimizer='adam',
    # loss=tf.keras.losses.mean_squared_error(),
    loss='mse',
    metrics=['mse', 'mae', 'accuracy']
)

EPOCHS = 10
history = model.fit(
    x = train_data,
    y = train_label,
    validation_split=0.1,
    batch_size=32,
    epochs=EPOCHS
)

# print(history)

# fig, loss_ax = plt.subplots()
# acc_ax = loss_ax.twinx()

# loss_ax.plot(history.history['loss'], 'y', label='train loss')
# loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# loss_ax.legend(loc='upper left')

# acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
# acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
# acc_ax.set_ylabel('accuracy')
# acc_ax.legend(loc='lower left')

#plt.show()

#model.save('D:\\example_STFT.h5')


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('D:\\test_tflite_STFT.tflite', 'wb').write(tflite_model)


##
# if __name__ == '__main__':
#     print("hello, world~!!")




## endl
