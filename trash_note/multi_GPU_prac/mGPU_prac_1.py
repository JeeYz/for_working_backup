import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers.experimental import preprocessing


train_data_file = "D:\\train_data_.npz"

loaded_data = np.load(train_data_file)

# temp_train_data = loaded_data['data']
# train_label = loaded_data['label']

temp_train_data = loaded_data['data'][:10]
train_label = loaded_data['label'][:10]

temp_train_data = tf.data.Dataset.from_tensor_slices(temp_train_data)
train_label = tf.data.Dataset.from_tensor_slices(train_label)

# temp_train_data = tf.data.Dataset.from_generator(temp_train_data)
# train_label = tf.data.Dataset.from_generator(train_label)

input_sigs = keras.Input(shape=(64000,))

@tf.function
def stft_func(input_sig, **kwargs):
    with tf.GradientTape() as tape:
        frame_size = 255
        delay_size = 128

        gap_sizes = frame_size - delay_size
        range_len = len(input_sig)//(frame_size-delay_size)

    for num in range(range_len):

        one_data = input_sig[num*gap_sizes:(num+1)*gap_sizes]

        x = tf.cast(one_data, tf.complex64)
        tres = tf.raw_ops.FFT(input=x)
        
        x = tf.abs(tres)
        x = tf.cast(x, tf.float32)
        x = tf.expand_dims(x, 0)
        if num == 0:
            a = tf.zeros(x.shape)
            result = tf.raw_ops.Add(x=a, y=x)
        else:
            # print(result.shape)
            result = tf.concat([result, x], axis=0)
        return result

print(input_sigs)

for i, one_data in enumerate(input_sigs.numpy):

    result = Lambda(stft_func)(one_data)
    # result = stft_func(one_data)
    result = tf.expand_dims(result, 0)

    if i == 0:
        a = tf.zeros(result.shape)
        train_data = tf.raw_ops.Add(x=a, y=result)
    else:
        train_data = tf.concat([train_data, result], axis=0)

    print("\r{}th file is done...".format(i+1), end='')



# for i, one_data in enumerate(input_sig):

#     # result = Lambda(stft_func)(one_data)
#     frame_size = 255
#     delay_size = 128

#     print("******", one_data.shape)

#     gap_sizes = frame_size - delay_size
#     # range_len = len(input_sig)//(frame_size-delay_size)
#     range_len = one_data.shape[0]//(frame_size-delay_size)

#     for num in range(range_len):

#         t_one_data = one_data[num*gap_sizes:(num+1)*gap_sizes]

#         x = tf.cast(t_one_data, tf.complex64)
#         tres = tf.raw_ops.FFT(input=x)
        
#         x = tf.abs(tres)
#         x = tf.cast(x, tf.float32)
#         x = tf.expand_dims(x, 0)
#         if num == 0:
#             a = tf.zeros(x.shape)
#             result = tf.raw_ops.Add(x=a, y=x)
#         else:
#             # print(result.shape)
#             result = tf.concat([result, x], axis=0)

#     result = tf.expand_dims(result, 0)

#     if i == 0:
#         a = tf.zeros(result.shape)
#         train_data = tf.raw_ops.Add(x=a, y=result)
#     else:
#         train_data = tf.concat([train_data, result], axis=0)

#     print("\r{}th file is done...".format(i+1), end='')




result = np.expand_dims(train_data, -1)
# result = tf.raw_ops.ExpandDims(train_data, -1)

print(result.shape)


x = preprocessing.Resizing(32, 32)(result)
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

model = keras.Model(inputs=input_sigss, outputs=answer)

model.summary()

keras.utils.plot_model(model, "proto_model_with_shape_info.png", show_shapes=True)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

history = model.fit(x = temp_train_data, y = train_label, batch_size=64, epochs=10, validation_split=0.2)


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('D:\\test_tflite_mGPU_prac_1.tflite', 'wb').write(tflite_model)

