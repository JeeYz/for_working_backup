from shutil import SameFileError
import tensorflow as tf
import wavio
import numpy as np
import scipy.signal as sps

WAV_FILENAME = "59183529_7b8aee0989_25.wav"
SEG_DATA_SEC = 0.005
SAMPLERATE_LENGTH = 16000
SEG_DATA_LENGTH = SAMPLERATE_LENGTH*SEG_DATA_SEC


a = wavio.read(WAV_FILENAME)
print(a)

data = a.data[:, 0]
print(data)

curr_data = (data-np.mean(data))/np.std(data)
curr_data = sps.decimate(curr_data, 3)
curr_data = np.array(curr_data, dtype=np.float32)

# temp_label = np.random.randint(10, size=(800, 1, 1))
temp_label = np.random.randint(10, size=(1, 800, 1))

print(temp_label.shape)

print(curr_data)
print(len(curr_data))


def devide_data(input_data):
    result = list()
    temp = list()

    for i,one_data in enumerate(input_data):
        if (i+1)%SEG_DATA_LENGTH != 0:
            temp.append(one_data)
        else:
            temp.append(one_data)
            temp = np.array(temp, dtype=np.float32)
            result.append(temp)
            temp = list()

    return result


curr_data = np.array([devide_data(curr_data)], dtype=np.float32)
print(curr_data.shape)
# curr_data = curr_data.transpose(1,0,2)
# print(curr_data.shape)


input = tf.keras.Input(shape=(800, 80))

x = tf.keras.layers.LSTM(16, return_sequences=True)(input)
output = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(input, output)

model.compile(loss=tf.nn.ctc_loss, optimizer='adam',
              metrics=['sparse_categorical_accuracy'])
model.fit(curr_data, temp_label, epochs=10)







