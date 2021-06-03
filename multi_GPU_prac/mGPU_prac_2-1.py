import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_io as tfio

from tensorflow.keras.layers.experimental import preprocessing

import numpy as np

train_data_file = "D:\\train_data_.npz"

loaded_data = np.load(train_data_file)

train_data_1 = loaded_data['data']
train_label_1 = loaded_data['label']

norm_audio_numpy = (train_data_1-np.min(train_data_1))/(np.max(train_data_1)-np.min(train_data_1))

train_data = tf.data.Dataset.from_tensor_slices((norm_audio_numpy, train_label_1)).shuffle(5000).batch(64)

import time

input_sig = tf.keras.Input(shape=(64000,))

x = tf.signal.stft(input_sig, frame_length=255, frame_step=128)

x = tf.abs(x)

x = tf.expand_dims(x, -1)

print(x)

x = preprocessing.Resizing(32, 32)(x)

x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.2)(x)
answer = layers.Dense(7, activation='softmax')(x)

# print(answer)

model = tf.keras.Model(inputs=input_sig, outputs=answer)

model.summary()

tf.keras.utils.plot_model(model, "proto_model_with_shape_info.png", show_shapes=True)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

history = model.fit(train_data, epochs=10)




converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('D:\\test_tflite_mGPU_prac_2-1.tflite', 'wb').write(tflite_model)