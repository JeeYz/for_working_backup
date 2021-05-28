import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python import framework
from tensorflow.python.framework.tensor_util import constant_value
from tensorflow.keras.layers import Lambda


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

        if max_num > len(one_data):
            one_data = np.pad(one_data, (0, (max_num-len(one_data))))

        x = tf.cast(one_data, tf.complex64)
        tres = tf.signal.fft(x)
        
        x = tf.abs(tres)
        x = tf.cast(x, tf.float32)
        result.append(x)
    return np.array(result)


train_data = list()

train_data_file = "D:\\train_data_.npz"

loaded_data = np.load(train_data_file)

temp_train_data = loaded_data['data']
train_label = loaded_data['label']

# temp_train_data = loaded_data['data'][:10]
# train_label = loaded_data['label'][:10]

# temp_train_data = tf.data.Dataset.from_tensor_slices(temp_train_data)
# train_label = tf.data.Dataset.from_tensor_slices(train_label)

# temp_train_data = tf.data.Dataset.from_generator(temp_train_data)
# train_label = tf.data.Dataset.from_generator(train_label)

# input_sig = keras.Input(shape=(64000))

for i, one_data in enumerate(temp_train_data):
    result = stft_func(one_data, frame_size=255, delay_size=128)
    # # result = Lambda(stft_func)(one_data, frame_size=255, delay_size=128)
    # result = Lambda(stft_func, arguments={'frame_size':255, 'delay_size':128})(one_data)
    train_data.append(result)
    print("\r{}th file is done...".format(i+1), end='')

train_data = np.array(train_data)

# result = np.expand_dims(train_data, -1)
result = tf.expand_dims(train_data, -1)
print(result.shape)

input_data = tf.data.Dataset.from_tensor_slices(result)
print(input_data)

input_sig = keras.Input(shape=(504, 127, 1))

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
answer = layers.Dense(7, activation='softmax')(x)


model = keras.Model(inputs=input_sig, outputs=answer)

model.summary()

keras.utils.plot_model(model, "proto_model_with_shape_info.png", show_shapes=True)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

history = model.fit(x = train_data, y = train_label, batch_size=64, epochs=10, validation_split=0.2)
