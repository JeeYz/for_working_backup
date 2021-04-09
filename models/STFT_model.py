
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
train_label = train_data_np['label']


train_label = np.expand_dims(train_label, axis=-1)
print("\n******************************\n")
print(train_label.shape)
print(train_label)
print("\n******************************\n")

print("++shape : {shape}".format(shape=train_data.shape))


input_vec = tf.keras.Input(shape=(32000,))
#@tf.function
#def extract_feature(input_data):
#    spectrogram = tf.signal.stft(input_data, frame_length=255, frame_step=128)
#    spectrogram = tf.abs(spectrogram)
#    spectrogram = tf.expand_dims(spectrogram, -1)
#    return spectrogram


#
#input_vec = tf.keras.Input(shape=(32000,))


##
#@tf.function
#def extract_feature(input_data):
#    spectrogram = tf.signal.stft(input_data, frame_length=255, frame_step=128)
#    spectrogram = tf.abs(spectrogram)
#    spectrogram = tf.expand_dims(spectrogram, -1)
#    return spectrogram

# spectrogram = np.asarray(spectrogram)
#spectrogram = tf.stack(spectrogram)
#input_vec = tf.stack(input_vec)
#spectrogram = extract_feature(train_data)


#spectrogram = tf.signal.stft(input_vec, frame_length=255, frame_step=128)
#spectrogram = tf.abs(spectrogram)
#spectrogram = tf.expand_dims(spectrogram, -1)



#spectrogram = tf.keras.layers.Lambda(extract_feature, name='extract_feature')(input_vec)
#
#
#
#print("\n******************************\n")
#print(spectrogram.shape)
#print(spectrogram)
#print("\n******************************\n")

x = tf.abs(input_vec)
x = tf.expand_dims(x, -1)
x = layers.Conv1D(32, 3, activation='relu')(x)
x = layers.Conv1D(64, 3, activation='relu')(x)

spectrogram = tf.expand_dims(x, -1)

# input_vec = tf.keras.Input(shape=(spectrogram.shape[1], spectrogram.shape[2], 1))
# x = preprocessing.Resizing(64, 64)(input_vec)
x = preprocessing.Resizing(32, 32)(spectrogram)
x = preprocessing.Normalization()(x)
# x = preprocessing.Normalization()(x)
# x = layers.Conv2D(32, 3, activation='relu')(input_vec)
x = layers.Conv2D(32, 3, activation='relu')(x)
# x = layers.Conv2D(32, 3, activation='relu')(x)
# x = layers.MaxPooling2D()(x)
# x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
answer = layers.Dense(6, activation='softmax')(x)

model = tf.keras.Model(inputs=input_vec, outputs=answer)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 1
history = model.fit(
    x = train_data,
    y = train_label,
    validation_split=0.2,
    batch_size=32,
    epochs=EPOCHS
)

print(history)

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='lower left')

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
