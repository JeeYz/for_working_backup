#%%
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# train_data_path = "D:\\train_data_.npz"
# train_data_path = "D:\\train_data_mid_.npz"
# train_data_path = "D:\\train_data_small_2sec_.npz"
# train_data_path = "D:\\train_data_mid_2sec_.npz"
# train_data_path = "D:\\train_data_mid_2sec_backup_.npz"
# train_data_path = "D:\\train_data_mid_2sec_random_.npz"
# train_data_path = "D:\\train_data_small_2sec_random_.npz"
train_data_path = "D:\\train_data_small_16000_random_.npz"
# train_data_path = "D:\\train_data_00.npz"


# test_data_path = "D:\\test_data_.npz"
# test_data_path = "D:\\test_data_2sec_.npz"
test_data_path = "D:\\test_data_16000_.npz"


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
)).shuffle(5000).batch(128)
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

tensor_slice_size = 128
tensor_shift_size = 64
tensor_gap_size = tensor_slice_size-tensor_shift_size

cnn_chan_size = 64


with mirrored_strategy.scope():

    input_sig = tf.keras.Input(shape=(16000,))

    x = tf.signal.stft(input_sig, frame_length=255, frame_step=128)

    x = tf.abs(x)
    
    x = tf.expand_dims(x, -1)
           
    x = preprocessing.Resizing(32, 32)(x) 
    x = layers.BatchNormalization()(x)
    
    cnn_block_0 = CNN_block(channel_size=cnn_chan_size)
    
    x = cnn_block_0(x)

    x = layers.Flatten()(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    answer = layers.Dense(6, activation='softmax')(x)

    # print(answer)

    model = tf.keras.Model(inputs=input_sig, outputs=answer)

    model.summary()

    tf.keras.utils.plot_model(model, "proto_model_with_shape_info_cnn_0.png", show_shapes=True)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

#%%
history = model.fit(train_dataset, epochs=12)

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
open('D:\\test_tflite_file_cnn_0.tflite', 'wb').write(tflite_model)


print('\n\n')
print("loss : {}, accuracy : {}".format(eval_loss, eval_acc))
print('\n\n')


commands=['camera', 'picture', 'record', 'stop', 'end', 'None']

y_pred = np.argmax(model.predict(test_data_00), axis=1)
y_true = test_label_00

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
# sns.heatmap(confusion_mtx, xticklables=commands, yticklabels=commands, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

