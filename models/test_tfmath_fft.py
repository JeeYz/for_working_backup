import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

sample_rate_size = 16000
recording_time = 2

train_data_size = 100

test_label_size = 2

a = np.random.randint(-100, 100, size=(10))

full_train_data = list()
full_train_label = list()

for i in range(train_data_size):
    a = np.random.randint(-1000, 1000, size=(sample_rate_size*recording_time))
    full_train_data.append(a)

full_train_data = np.array(full_train_data)

for i in range(train_data_size):
    a = np.random.randint(0, 2)
    full_train_label.append(a)

full_train_label = np.array(full_train_label)

# full_train_data = tf.cast(full_train_data, tf.float32)
# full_train_label = tf.cast(full_train_label, tf.float32)

input_vec = tf.keras.Input(shape=(32000,))

x = tf.cast(input_vec, tf.complex64)

x = tf.signal.fft(x)
x = tf.abs(x)
x = tf.cast(x, tf.float32)
x = tf.reshape(x, (-1, 320, 100))
x = tf.expand_dims(x, -1)

x = preprocessing.Resizing(32, 32)(x)
x = preprocessing.Normalization()(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
answer = layers.Dense(2, activation='softmax')(x)


model = tf.keras.Model(inputs=input_vec, outputs=answer)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    x = full_train_data,
    y = full_train_label,
    batch_size=32,
    epochs=10
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('D:\\test_tflite_STFT.tflite', 'wb').write(tflite_model)

