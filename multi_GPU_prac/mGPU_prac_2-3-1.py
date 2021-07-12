#%%
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np

# train_data_path = "D:\\train_data_.npz"
# train_data_path = "D:\\train_data_mid_.npz"
# train_data_path = "D:\\train_data_small_2sec_.npz"
train_data_path = "D:\\train_data_mid_2sec_.npz"
# train_data_path = "D:\\train_data_00.npz"


# test_data_path = "D:\\test_data_.npz"
test_data_path = "D:\\test_data_2sec_.npz"


loaded_data_00 = np.load(train_data_path)
train_data_00 = loaded_data_00['data']
train_label_00 = loaded_data_00['label']

loaded_data_01 = np.load(test_data_path)
test_data_00 = loaded_data_01['data']
test_label_00 = loaded_data_01['label']



#%% class -> residual cnn block
class residual_cnn_block_2D(layers.Layer):

    def __init__(self, **kwarg):
        super(residual_cnn_block_2D, self).__init__()

        if "channel_size" in kwarg.keys():
            self.chan_size = kwarg['channel_size']


    def __call__(self, inputs, **kwarg):
        conv2d_layer_1 = layers.Conv2D(self.chan_size[0], (3, 3),
                                       padding='same')
        conv2d_layer_2 = layers.Conv2D(self.chan_size[1], (3, 3),
                                       padding='same')

        init_val = inputs
        # inputs = layers.BatchNormalization()(inputs)

        x = conv2d_layer_1(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_2(x)
        x = layers.BatchNormalization()(x)
        # print('*********', self.chan_size)
        y = layers.Conv2D(self.chan_size[1], (1, 1), padding='same')(init_val)
        x = tf.math.add(y, x)
        x = tf.nn.relu(x)
        # x = layers.MaxPooling2D(pool_size=(2, 1), padding='same')(x)
        x = layers.Dropout(0.2)(x)

        return x





#%% 
class CNN_block(layers.Layer):

    def __init__(self, **kwarg):
        super(CNN_block, self).__init__()

        if "channel_size" in kwarg.keys():
            self.chan_size = kwarg['channel_size']


    def __call__(self, inputs, **kwarg):
        conv2d_layer_1 = layers.Conv2D(self.chan_size, (3, 3),
                                       padding='same')
        conv2d_layer_2 = layers.Conv2D(self.chan_size*2, (3, 3),
                                       padding='same')
        conv2d_layer_3 = layers.Conv2D(self.chan_size*3, (3, 3),
                                       padding='same')
        conv2d_layer_4 = layers.Conv2D(self.chan_size*4, (3, 3),
                                       padding='same')

        
        inputs = layers.BatchNormalization()(inputs)

        x = conv2d_layer_1(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_2(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_3(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_4(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        return x






#%%
def generate_train_data():

    for one_data, one_label in zip(train_data_00, train_label_00):
        yield (one_data, one_label)


def generate_test_data():
    for one_data, one_label in zip(test_data_00, test_label_00):
        yield (one_data, one_label)
    
    # yield (train_data_00, train_label_00)

#%%
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

#%%
train_dataset = tf.data.Dataset.from_generator(generate_train_data, 
# output_types=(tf.float32, tf.int32),
output_signature=(
                    tf.TensorSpec(shape=(None,), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int32)
)).shuffle(5000).batch(256)
# args=(train_data_path)),
train_dataset = train_dataset.with_options(options)

# train_dataset = train_dataset.cache()

#%%
test_dataset = tf.data.Dataset.from_generator(generate_test_data, 
# output_types=(tf.float32, tf.int32),
output_signature=(
                    tf.TensorSpec(shape=(None,), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int32)
)).shuffle(5000).batch(64)
# args=(test_data_path))
test_dataset = test_dataset.with_options(options)

# test_dataset = train_dataset.cache()

#%%
mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

tensor_slice_size = 64
tensor_shift_size = 32
tensor_gap_size = tensor_slice_size-tensor_shift_size

chan_size = 64


with mirrored_strategy.scope():

    input_sig = tf.keras.Input(shape=(20000,))

    x = tf.signal.stft(input_sig, frame_length=255, frame_step=128)

    x = tf.abs(x)

    tmp_shape = x.shape
    print('\n\n\n')
    print(tmp_shape)
    print('\n\n\n')

    temp_size = 0
    x0 = tf.slice(  x, begin=[0, temp_size,0], 
                    size=[  -1, 
                            tensor_slice_size, tmp_shape[-1]])

    temp_size += tensor_gap_size
    x1 = tf.slice(  x, begin=[0, temp_size,0], 
                    size=[  -1, 
                            tensor_slice_size, tmp_shape[-1]])

    temp_size += tensor_gap_size
    x2 = tf.slice(  x, begin=[0, temp_size,0], 
                    size=[  -1, 
                            tensor_slice_size, tmp_shape[-1]])

    temp_size += tensor_gap_size
    x3 = tf.slice(  x, begin=[0, temp_size,0], 
                    size=[  -1, 
                            59, tmp_shape[-1]])

    # temp_size += tensor_gap_size
    # x4 = tf.slice(  x, begin=[0, temp_size,0], 
    #                 size=[  -1, 
    #                         tensor_slice_size, tmp_shape[-1]])

    # temp_size += tensor_gap_size
    # x5 = tf.slice(  x, begin=[0, temp_size,0], 
    #                 size=[  -1, 
    #                         tensor_slice_size, tmp_shape[-1]])

    # temp_size += tensor_gap_size
    # x6 = tf.slice(  x, begin=[0, temp_size,0], 
    #                 size=[  -1, 
    #                         tensor_slice_size-(512-tmp_shape[-2]), 
    #                         tmp_shape[-1]])

    temp_size += tensor_gap_size
    x7 = tf.slice(  x, begin=[0, temp_size,0], 
                    size=[  -1, 
                            tmp_shape[-2]-temp_size, 
                            tmp_shape[-1]])

    
    x0 = tf.expand_dims(x0, -1)
    x1 = tf.expand_dims(x1, -1)
    x2 = tf.expand_dims(x2, -1)
    x3 = tf.expand_dims(x3, -1)
    # x4 = tf.expand_dims(x4, -1)
    # x5 = tf.expand_dims(x5, -1)
    # x6 = tf.expand_dims(x6, -1)
    x7 = tf.expand_dims(x7, -1)

       
    x0 = preprocessing.Resizing(32, 64)(x0) 
    x0 = layers.BatchNormalization()(x0)

    x1 = preprocessing.Resizing(32, 64)(x1) 
    x1 = layers.BatchNormalization()(x1)

    x2 = preprocessing.Resizing(32, 64)(x2) 
    x2 = layers.BatchNormalization()(x2)

    x3 = preprocessing.Resizing(32, 64)(x3) 
    x3 = layers.BatchNormalization()(x3)

    # x4 = preprocessing.Resizing(32, 32)(x4) 
    # x5 = preprocessing.Resizing(32, 32)(x5) 
    # x6 = preprocessing.Resizing(32, 32)(x6) 
    x7 = preprocessing.Resizing(32, 64)(x7) 
    x7 = layers.BatchNormalization()(x7) 



    cnn_block_0 = residual_cnn_block_2D(channel_size=[64, 64])
    cnn_block_1 = residual_cnn_block_2D(channel_size=[128, 128])

    cnn_block_2 = residual_cnn_block_2D(channel_size=[64, 64])
    cnn_block_3 = residual_cnn_block_2D(channel_size=[128, 128])

    cnn_block_4 = residual_cnn_block_2D(channel_size=[64, 64])
    cnn_block_5 = residual_cnn_block_2D(channel_size=[128, 128])

    cnn_block_6 = residual_cnn_block_2D(channel_size=[64, 64])
    cnn_block_7 = residual_cnn_block_2D(channel_size=[128, 128])

    # cnn_block_8 = residual_cnn_block_2D(channel_size=[64, 64])
    # cnn_block_9 = residual_cnn_block_2D(channel_size=[128, 128])

    # cnn_block_10 = residual_cnn_block_2D(channel_size=[64, 64])
    # cnn_block_11 = residual_cnn_block_2D(channel_size=[128, 128])

    # cnn_block_12 = residual_cnn_block_2D(channel_size=[64, 64])
    # cnn_block_13 = residual_cnn_block_2D(channel_size=[128, 128])

    cnn_block_14 = residual_cnn_block_2D(channel_size=[64, 64])
    cnn_block_15 = residual_cnn_block_2D(channel_size=[128, 128])

    x0 = cnn_block_0(x0)
    x0 = layers.BatchNormalization()(x0)
    x0 = cnn_block_1(x0)
    x0 = layers.BatchNormalization()(x0)

    x1 = cnn_block_2(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = cnn_block_3(x1)
    x1 = layers.BatchNormalization()(x1)

    x2 = cnn_block_4(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = cnn_block_5(x2)
    x2 = layers.BatchNormalization()(x2)

    x3 = cnn_block_6(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = cnn_block_7(x3)
    x3 = layers.BatchNormalization()(x3)

    # x4 = cnn_block_0(x4)
    # x4 = cnn_block_1(x4)

    # x5 = cnn_block_0(x5)
    # x5 = cnn_block_1(x5)

    # x6 = cnn_block_0(x6)
    # x6 = cnn_block_1(x6)

    x7 = cnn_block_14(x7)
    x7 = layers.BatchNormalization()(x7)
    x7 = cnn_block_15(x7)
    x7 = layers.BatchNormalization()(x7)


    x0 = layers.Flatten()(x0)
    x1 = layers.Flatten()(x1)
    x2 = layers.Flatten()(x2)
    x3 = layers.Flatten()(x3)
    # x4 = layers.Flatten()(x4)
    # x5 = layers.Flatten()(x5)
    # x6 = layers.Flatten()(x6)
    x7 = layers.Flatten()(x7)

    x = tf.concat([ x0, x1, x2, x3, 
                    # x4, x5, x6, 
                    x7], -1)
    # x = tf.concat([x0, x1, x2, x3, x4, x5, x6], -1)

    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.25)(x)
    # x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    answer = layers.Dense(6, activation='softmax')(x)

    # print(answer)

    model = tf.keras.Model(inputs=input_sig, outputs=answer)

    model.summary()

    tf.keras.utils.plot_model(model, "proto_model_with_shape_info_resnet_1.png", show_shapes=True)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

#%%
history = model.fit(train_dataset, epochs=15)

#%%
eval_loss, eval_acc = model.evaluate(test_dataset)

print('\n\n')
print("loss : {}, accuracy : {}".format(eval_loss, eval_acc))
print('\n\n')


#%%
## convert tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('D:\\test_tflite_file.tflite_rnet_1_slice', 'wb').write(tflite_model)

print('\n\n')
print("loss : {}, accuracy : {}".format(eval_loss, eval_acc))
print('\n\n')
