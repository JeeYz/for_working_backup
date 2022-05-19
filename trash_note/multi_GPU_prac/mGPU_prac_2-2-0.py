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
# train_data_path = "D:\\train_data_small_16000_random_.npz"
# train_data_path = "D:\\train_data_big_16000_random_.npz"
# train_data_path = "D:\\train_data_small_20000_random_.npz"
# train_data_path = "D:\\train_data_mini_20000_random_.npz"
# train_data_path = "D:\\train_data_middle_20000_random_.npz"
train_data_path = "D:\\train_data_middle_20000_random_1_.npz"
# train_data_path = "D:\\train_data_middle_20000_random_2_.npz"
# train_data_path = "D:\\train_data_middle_20000_random_confirm_0_.npz"
# train_data_path = "D:\\train_data_00.npz"


# test_data_path = "D:\\test_data_.npz"
# test_data_path = "D:\\test_data_2sec_.npz"
# test_data_path = "D:\\test_data_16000_.npz"
# test_data_path = "D:\\test_data_mini_16000_.npz"
test_data_path = "D:\\test_data_20000_.npz"
# test_data_path = "D:\\test_data_20000_confirm_0_.npz"

loaded_data_00 = np.load(train_data_path)
train_data_00 = loaded_data_00['data']
train_label_00 = loaded_data_00['label']

loaded_data_01 = np.load(test_data_path)
test_data_00 = loaded_data_01['data']
test_label_00 = loaded_data_01['label']

print(test_label_00.shape)

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
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_2(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_3(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_4(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        return x
    
    
    

#%%
class CNN_block_deep(layers.Layer):

    def __init__(self, **kwarg):
        super(CNN_block_deep, self).__init__()

        if "channel_size" in kwarg.keys():
            self.chan_size = kwarg['channel_size']


    def __call__(self, inputs, **kwarg):
        conv2d_layer_1 = layers.Conv2D(self.chan_size, (3, 3),
                                       padding='same')
        conv2d_layer_2 = layers.Conv2D(self.chan_size, (3, 3),
                                       padding='same')
        conv2d_layer_3 = layers.Conv2D(self.chan_size, (3, 3),
                                       padding='same')
        conv2d_layer_4 = layers.Conv2D(self.chan_size, (3, 3),
                                       padding='same')

        
        inputs = layers.BatchNormalization()(inputs)

        x = conv2d_layer_1(inputs)
        # x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_2(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_3(x)
        # x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_4(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
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

tensor_slice_size = 128
tensor_shift_size = 64
tensor_gap_size = tensor_slice_size-tensor_shift_size

cnn_chan_size = 128


with mirrored_strategy.scope():

    input_sig = tf.keras.Input(shape=(20000,))

    x = tf.signal.stft(input_sig, frame_length=255, frame_step=128)

    # x = tf.math.log(x)

    x = tf.abs(x)
        
    x = tf.expand_dims(x, -1)
           
    # x = preprocessing.Resizing(32, 32)(x)
    x = layers.BatchNormalization()(x)
    
    # cnn_block_0 = CNN_block(channel_size=cnn_chan_size)
    # cnn_block_1 = CNN_block(channel_size=cnn_chan_size*4)
    
    cnn_block_0 = CNN_block_deep(channel_size=cnn_chan_size)
    cnn_block_1 = CNN_block_deep(channel_size=cnn_chan_size*2)
    cnn_block_2 = CNN_block_deep(channel_size=cnn_chan_size*3)
    cnn_block_3 = CNN_block_deep(channel_size=cnn_chan_size*4)
    cnn_block_4 = CNN_block_deep(channel_size=cnn_chan_size*5)
    cnn_block_5 = CNN_block_deep(channel_size=cnn_chan_size*6)
    cnn_block_6 = CNN_block_deep(channel_size=cnn_chan_size*7)
    
    # x = cnn_block_0(x)
    # x = cnn_block_1(x)
    # x = cnn_block_2(x)
    # x = cnn_block_3(x)
    # x = cnn_block_4(x)
    # x = cnn_block_1(x)
    
    
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.MaxPooling2D((4, 4), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = layers.BatchNormalization()(x)
        
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.MaxPooling2D((4, 4), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    # x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = layers.BatchNormalization()(x)
    

    x = layers.Flatten()(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(64)(x)
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
model_class_weights = {
                        0: 1.0,
                        1: 1.0,
                        2: 1.0,
                        3: 1.0,
                        4: 1.0,
                        5: 0.5
                           }

history = model.fit(train_dataset, epochs=10,
                    class_weight=model_class_weights,
                    )

#%%
eval_loss, eval_acc = model.evaluate(test_dataset)

print('\n\n')
print("loss : {}, accuracy : {}".format(eval_loss, eval_acc))
print('\n\n')





#%%
# convert tflite

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()
# open('D:\\test_tflite_file_cnn_0.tflite', 'wb').write(tflite_model)


# print('\n\n')
# print("loss : {}, accuracy : {}".format(eval_loss, eval_acc))
# print('\n\n')


# commands=['camera', 'picture', 'record', 'stop', 'end', 'None']

# y_pred = np.argmax(model.predict(test_data_00), axis=1)
# y_true = test_label_00

# confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(10,8))
# sns.heatmap(confusion_mtx, xticklables=commands, yticklabels=commands, annot=True, fmt='g')
# plt.xlabel('Prediction')
# plt.ylabel('Label')
# plt.show()




#%%
# test_tensor = tf.data.Dataset.from_tensors(test_data_00)




#%%
result_of_predict = model.predict(test_data_00)
result_arg = tf.math.argmax(result_of_predict, 1).numpy()

result_of_predict_train_data = model.predict(train_data_00)
result_arg_train_data = tf.math.argmax(result_of_predict_train_data, 1).numpy()




#%% F1 Score 계산하기
labels_num = 6

confusion_matrix = np.zeros((labels_num, labels_num), dtype=np.int16)

length_of_true_ans = len(test_label_00)

for i in range(length_of_true_ans):
    x_axis = int(test_label_00[i])
    y_axis = int(result_arg[i])
    confusion_matrix[x_axis, y_axis]+=1


print('\n')
print(confusion_matrix)
print('\n')

sum_ax_0 = np.sum(confusion_matrix, axis=0)
sum_ax_1 = np.sum(confusion_matrix, axis=1)

print('\n')
print(sum_ax_0)
print(sum_ax_1)
print('\n')

precisions = list()
recalls = list()

for i in range(labels_num):

    if sum_ax_0[i] != 0:
        temp_val = confusion_matrix[i][i]/sum_ax_0[i]
        precisions.append(temp_val)
    else:
        precisions.append(0.0)

    if sum_ax_1[i] != 0:
        temp_val = confusion_matrix[i][i]/sum_ax_1[i]
        recalls.append(temp_val)
    else:
        precisions.append(0.0)


print('\n')
for i in range(labels_num):
    print("{} label : precision : {}, recall : {}".format(i, precisions[i], recalls[i]))

print('\n')

temp_sum = 0
for i in range(labels_num):
    temp_sum+=confusion_matrix[i][i]

total_acc = temp_sum/np.sum(sum_ax_0)
total_acc_1 = temp_sum/len(test_label_00)

print("number of test data : {a}".format(a=len(test_label_00)))
print("sum of right answers : {b}".format(b=temp_sum))
print("sum of sum_ax_0 : {}".format(np.sum(sum_ax_0)))
print("sum of sum_ax_1 : {}".format(np.sum(sum_ax_1)))
print("accuracy : {}".format(total_acc))
print("accuracy 1 : {}".format(total_acc_1))




#%% F1 Score 계산하기
labels_num_td = 6

confusion_matrix = np.zeros((labels_num_td, labels_num_td), dtype=np.int32)

length_of_true_ans = len(train_label_00)
print('\n\n')
print(length_of_true_ans)
print(len(result_arg_train_data))
print(len(train_label_00))
print(len(train_data_00))

temp_num = 0

for i in range(length_of_true_ans):
    x_axis = int(train_label_00[i])
    y_axis = int(result_arg_train_data[i])
    if x_axis < 0 or y_axis < 0:
        temp_num+=1
    confusion_matrix[x_axis, y_axis]+=1

print(temp_num)


print('\n')
print(confusion_matrix)
print('\n')

sum_ax_0 = np.sum(confusion_matrix, axis=0)
sum_ax_1 = np.sum(confusion_matrix, axis=1)

print('\n')
print(sum_ax_0)
print(sum_ax_1)
print('\n')

precisions = list()
recalls = list()

for i in range(labels_num_td):

    if sum_ax_0[i] != 0:
        temp_val = confusion_matrix[i][i]/sum_ax_0[i]
        precisions.append(temp_val)
    else:
        precisions.append(0.0)

    if sum_ax_1[i] != 0:
        temp_val = confusion_matrix[i][i]/sum_ax_1[i]
        recalls.append(temp_val)
    else:
        precisions.append(0.0)


print('\n')
for i in range(labels_num_td):
    print("{} label : precision : {}, recall : {}".format(i, precisions[i], recalls[i]))

print('\n')

temp_sum = 0
for i in range(labels_num_td):
    temp_sum+=confusion_matrix[i][i]

total_acc = temp_sum/np.sum(sum_ax_0)
total_acc_1 = temp_sum/len(train_label_00)

print("number of test data : {a}".format(a=len(train_label_00)))
print("sum of right answers : {b}".format(b=temp_sum))
print("sum of sum_ax_0 : {}".format(np.sum(sum_ax_0)))
print("sum of sum_ax_1 : {}".format(np.sum(sum_ax_1)))
print("accuracy : {}".format(total_acc))
print("accuracy 1 : {}".format(total_acc_1))




#%%
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
# loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['accuracy'], 'b', label='train accuracy')
# acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()




