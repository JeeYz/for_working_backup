import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np


train_data_path = "D:\\"
train_file_list = [ "train_data_00.npz", 
                    "train_data_01.npz",
                    "train_data_02.npz",
                    "train_data_03.npz",
                    "train_data_04.npz",
                    "train_data_05.npz",
                    "train_data_06.npz",
                    "train_data_07.npz",
                    "train_data_08.npz",
                    "train_data_09.npz",
                    "train_data_10.npz",
                    "train_data_11.npz",
                    "train_data_12.npz",
                    "train_data_13.npz",
                    "train_data_14.npz"]


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

loaded_data_00 = np.load(train_data_path+train_file_list[0])
train_data_00 = loaded_data_00['data']
train_label_00 = loaded_data_00['label']
train_dataset_00 = tf.data.Dataset.from_tensor_slices((train_data_00, train_label_00)).shuffle(5000).batch(64)
train_dataset_00 = train_dataset_00.with_options(options)

loaded_data_01 = np.load(train_data_path+train_file_list[1])
train_data_01 = loaded_data_01['data']
train_label_01 = loaded_data_01['label']
train_dataset_01 = tf.data.Dataset.from_tensor_slices((train_data_01, train_label_01)).shuffle(5000).batch(64)
train_dataset_01 = train_dataset_01.with_options(options)

loaded_data_02 = np.load(train_data_path+train_file_list[2])
train_data_02 = loaded_data_02['data']
train_label_02 = loaded_data_02['label']
train_dataset_02 = tf.data.Dataset.from_tensor_slices((train_data_02, train_label_02)).shuffle(5000).batch(64)
train_dataset_02 = train_dataset_02.with_options(options)

loaded_data_03 = np.load(train_data_path+train_file_list[3])
train_data_03 = loaded_data_03['data']
train_label_03 = loaded_data_03['label']
train_dataset_03 = tf.data.Dataset.from_tensor_slices((train_data_03, train_label_03)).shuffle(5000).batch(64)
train_dataset_03 = train_dataset_03.with_options(options)

loaded_data_04 = np.load(train_data_path+train_file_list[4])
train_data_04 = loaded_data_04['data']
train_label_04 = loaded_data_04['label']
train_dataset_04 = tf.data.Dataset.from_tensor_slices((train_data_04, train_label_04)).shuffle(5000).batch(64)
train_dataset_04 = train_dataset_04.with_options(options)

loaded_data_05 = np.load(train_data_path+train_file_list[5])
train_data_05 = loaded_data_05['data']
train_label_05 = loaded_data_05['label']
train_dataset_05 = tf.data.Dataset.from_tensor_slices((train_data_05, train_label_05)).shuffle(5000).batch(64)
train_dataset_05 = train_dataset_05.with_options(options)

loaded_data_06 = np.load(train_data_path+train_file_list[6])
train_data_06 = loaded_data_06['data']
train_label_06 = loaded_data_06['label']
train_dataset_06 = tf.data.Dataset.from_tensor_slices((train_data_06, train_label_06)).shuffle(5000).batch(64)
train_dataset_06 = train_dataset_06.with_options(options)

loaded_data_07 = np.load(train_data_path+train_file_list[7])
train_data_07 = loaded_data_07['data']
train_label_07 = loaded_data_07['label']
train_dataset_07 = tf.data.Dataset.from_tensor_slices((train_data_07, train_label_07)).shuffle(5000).batch(64)
train_dataset_07 = train_dataset_07.with_options(options)

loaded_data_08 = np.load(train_data_path+train_file_list[8])
train_data_08 = loaded_data_08['data']
train_label_08 = loaded_data_08['label']
train_dataset_08 = tf.data.Dataset.from_tensor_slices((train_data_08, train_label_08)).shuffle(5000).batch(64)
train_dataset_08 = train_dataset_08.with_options(options)

loaded_data_09 = np.load(train_data_path+train_file_list[9])
train_data_09 = loaded_data_09['data']
train_label_09 = loaded_data_09['label']
train_dataset_09 = tf.data.Dataset.from_tensor_slices((train_data_09, train_label_09)).shuffle(5000).batch(64)
train_dataset_09 = train_dataset_09.with_options(options)

loaded_data_10 = np.load(train_data_path+train_file_list[10])
train_data_10 = loaded_data_10['data']
train_label_10 = loaded_data_10['label']
train_dataset_10 = tf.data.Dataset.from_tensor_slices((train_data_10, train_label_10)).shuffle(5000).batch(64)
train_dataset_10 = train_dataset_10.with_options(options)

loaded_data_11 = np.load(train_data_path+train_file_list[11])
train_data_11 = loaded_data_11['data']
train_label_11 = loaded_data_11['label']
train_dataset_11 = tf.data.Dataset.from_tensor_slices((train_data_11, train_label_11)).shuffle(5000).batch(64)
train_dataset_11 = train_dataset_11.with_options(options)

loaded_data_12 = np.load(train_data_path+train_file_list[12])
train_data_12 = loaded_data_12['data']
train_label_12 = loaded_data_12['label']
train_dataset_12 = tf.data.Dataset.from_tensor_slices((train_data_12, train_label_12)).shuffle(5000).batch(64)
train_dataset_12 = train_dataset_12.with_options(options)

loaded_data_13 = np.load(train_data_path+train_file_list[13])
train_data_13 = loaded_data_13['data']
train_label_13 = loaded_data_13['label']
train_dataset_13 = tf.data.Dataset.from_tensor_slices((train_data_13, train_label_13)).shuffle(5000).batch(64)
train_dataset_13 = train_dataset_13.with_options(options)

loaded_data_14 = np.load(train_data_path+train_file_list[14])
train_data_14 = loaded_data_14['data']
train_label_14 = loaded_data_14['label']
train_dataset_14 = tf.data.Dataset.from_tensor_slices((train_data_14, train_label_14)).shuffle(5000).batch(64)
train_dataset_14 = train_dataset_14.with_options(options)



train_dataset_list = [  train_dataset_00,
                        train_dataset_01,
                        train_dataset_02,
                        train_dataset_03,
                        train_dataset_04,
                        train_dataset_05,
                        train_dataset_06,
                        train_dataset_07,
                        train_dataset_08,
                        train_dataset_09,
                        train_dataset_10,
                        train_dataset_11,
                        train_dataset_12,
                        train_dataset_13,
                        train_dataset_14]


loaded_data_test = np.load("D:\\test_data_.npz")
test_data = loaded_data_test['data']
test_label = loaded_data_test['label']
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label)).shuffle(5000).batch(1)
test_dataset = test_dataset.with_options(options)


import time

mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with mirrored_strategy.scope():

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


epochs_num = 10

for one_epoch in range(epochs_num):
    print('\n', "{}th epoch".format(one_epoch+1), '\n')
    for one_train_dataset in train_dataset_list:
        history = model.fit(one_train_dataset, epochs=1)

eval_loss, eval_acc = model.evaluate(test_dataset)

print('\n\n')
print("loss : {}, accuracy : {}".format(eval_loss, eval_acc))
print('\n\n')


## convert tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('D:\\test_tflite_mGPU_prac_2-1.tflite', 'wb').write(tflite_model)