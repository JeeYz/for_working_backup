#%%
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import copy

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'

# train_data_path = "D:\\train_data_small_20000_random_.npz"
# train_data_path = "D:\\train_data_mini_20000_random_.npz"
# train_data_path = "D:\\train_data_middle_20000_random_.npz"

train_data_path = "D:\\train_data_middle_20000_random_1_.npz"
# train_data_path = "D:\\train_data_middle_20000_random_2_.npz"
# train_data_path = "D:\\train_data_middle_20000_random_confirm_0_.npz"


test_data_path = "D:\\test_data_20000_.npz"

loaded_data_00 = np.load(train_data_path)
train_data_00 = loaded_data_00['data']
train_label_00 = loaded_data_00['label']

loaded_data_01 = np.load(test_data_path)
test_data_00 = loaded_data_01['data']
test_label_00 = loaded_data_01['label']



#%% class -> residual cnn block
class residual_block(layers.Layer):

    def __init__(self, ch):
        super(residual_block, self).__init__()
        self.ch_size = ch

    def __call__(self, input_x):
        conv2d_layer_1 = layers.Conv2D(self.ch_size, (3, 3), padding='same')
        conv2d_layer_2 = layers.Conv2D(self.ch_size, (3, 3), padding='same')

        init_val = copy.deepcopy(input_x)

        x = conv2d_layer_1(input_x)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = conv2d_layer_2(x)
        x = layers.BatchNormalization()(x)

        y = layers.Conv2D(self.ch_size, (1, 1), padding='same')(init_val)
        x = tf.math.add(y, x)
        x = tf.nn.relu(x)

        x = layers.Dropout(0.2)(x)

        return x



#%%
class residual_layers(layers.Layer):
    def __init__(self, num_of_layers):
        super(residual_layers, self).__init__()

        init_ch = 32
        
        self.layers_list = list()

        for i in range(1, (num_of_layers+1)):
            temp = residual_block(init_ch*i)
            self.layers_list.append(temp)


    def __call__(self, input_x):

        x = input_x

        for one_layer in self.layers_list:
            x = one_layer(x)

        return x



#%%
class autoencoder_cnn_block(layers.Layer):
    def __init__(self, **kwargs):
        super(autoencoder_cnn_block, self).__init__()

        self.encoder = tf.keras.Sequential([ 
                    layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2),
                    layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
                    layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=2),])

        self.decoder = tf.keras.Sequential([
                    layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
                    layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
                    layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
                    layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



#%%
class autoencoder_snn_block(layers.Layer):
    def __init__(self, **kwargs):
        super(autoencoder_snn_block, self).__init__()

        self.encoder = tf.keras.Sequential([ 
                    layers.Dense(512, activation='relu'),
                    layers.Dense(256, activation='relu'),
                    layers.Dense(128, activation='relu'),])

        self.decoder = tf.keras.Sequential([
                    layers.Dense(128, activation='relu'),
                    layers.Dense(256, activation='relu'),
                    layers.Dense(512, activation='relu'),
                    layers.Dense(768, activation='relu'),])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



#%% 실험 1-1
class experiment_models(layers.Layer):
    def __init__(self, **kwargs):
        super(experiment_models, self).__init__()

        if "experiment_num" in kwargs.keys():
            self.exper_num = kwargs["experiment_num"]
        else:
            self.exper_num = 1

        if "resize_bool" in kwargs.keys():
            self.resize_bool = kwargs["resize_bool"]
        else:
            self.resize_bool = False

        if "num_of_labels" in kwargs.keys():
            self.num_of_labels = kwargs["num_of_labels"]
        else:
            self.num_of_labels = 6

        self.init_ch_num = 32
        self.init_cells = 64

        self.dense_block = self.generate_dense_layers(2)
        self.conv2d_block = self.generate_conv_layers(5)
        self.lstm_block = self.generate_lstm_layers(3)
        self.bilstm_block = self.generate_bilstm_layers(3)
        self.residual_blocks = residual_layers(3)
        self.autoencoder_cnn_blocks = autoencoder_cnn_block()
        self.autoencoder_nn_blocks = autoencoder_snn_block()
        



    def generate_bilstm_layers(self, num_of_layers):
        layers_list = list()

        layers_list.append(layers.BatchNormalization())

        for i in range(num_of_layers):
            layers_list.append(layers.Bidirectional(layers.LSTM(self.init_cells, return_sequences=True)))
        
        layers_list.append(layers.Bidirectional(layers.LSTM(self.init_cells)))

        return tf.keras.Sequential(layers_list)



    def generate_lstm_layers(self, num_of_layers):
        layers_list = list()

        layers_list.append(layers.BatchNormalization())

        for i in range(num_of_layers):
            layers_list.append(layers.LSTM(self.init_cells, return_sequences=True))

        layers_list.append(layers.LSTM(self.init_cells))

        return tf.keras.Sequential(layers_list)



    def generate_conv_layers(self, num_of_layers):

        layers_list = list()

        layers_list.append(layers.BatchNormalization())

        for i in range(num_of_layers):
            layers_list.append(layers.Conv2D(   
                                                self.init_ch_num*(i+1), 
                                                (3, 3), 
                                                padding='same', 
                                                activation='relu', 
                                                strides=2,
                                            ))
            layers_list.append(layers.BatchNormalization())

        return tf.keras.Sequential(layers_list)



    def generate_dense_layers(self, num_of_layers):

        layers_list = list()

        layers_list.append(layers.BatchNormalization())

        for i in range(num_of_layers):
            layers_list.append(layers.Dense(256))
            layers_list.append(layers.Dropout(0.25))

        return tf.keras.Sequential(layers_list)



    def func_of_1(self, input_x):



        return result



    def func_of_2(self, input_x):

        

        return result



    def func_of_3(self, input_x):

        

        return result

        

    def __call__(self, input_x):

        

        x = tf.signal.stft(input_x, frame_length=255, frame_step=128)

        x = tf.abs(x)
            
        x = tf.expand_dims(x, -1)

        if self.resize_bool == True:    
            x = preprocessing.Resizing(32, 32)(x)
            x = layers.BatchNormalization()(x)
        
        
        
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
        result = layers.Dense(self.num_of_labels, activation='softmax')(x)

        return result




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
)).shuffle(5000).batch(16)
# args=(test_data_path))
test_dataset = test_dataset.with_options(options)

# test_dataset = train_dataset.cache()

#%%
mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

tensor_slice_size = 50
tensor_shift_size = 25
tensor_gap_size = tensor_slice_size-tensor_shift_size

cnn_chan_size = 64




#%%
with mirrored_strategy.scope():

    input_sig = tf.keras.Input(shape=(20000,))

    print(input_sig)
    print(input_sig.shape)
    
    x0 = tf.slice(input_sig, begin=[0, 0], size=[-1, 4000])
    x1 = tf.slice(input_sig, begin=[0, 2000], size=[-1, 4000])
    x2 = tf.slice(input_sig, begin=[0, 4000], size=[-1, 4000])
    x3 = tf.slice(input_sig, begin=[0, 6000], size=[-1, 4000])
    x4 = tf.slice(input_sig, begin=[0, 8000], size=[-1, 4000])
    x5 = tf.slice(input_sig, begin=[0, 10000], size=[-1, 4000])
    x6 = tf.slice(input_sig, begin=[0, 12000], size=[-1, 4000])
    x7 = tf.slice(input_sig, begin=[0, 14000], size=[-1, 4000])
    x8 = tf.slice(input_sig, begin=[0, 16000], size=[-1, 4000])

    cnn_layer_0 = CNN_block()
    cnn_layer_1 = CNN_block()
    cnn_layer_2 = CNN_block()
    cnn_layer_3 = CNN_block()
    cnn_layer_4 = CNN_block()
    cnn_layer_5 = CNN_block()
    cnn_layer_6 = CNN_block()
    cnn_layer_7 = CNN_block()
    cnn_layer_8 = CNN_block()

    x0 = cnn_layer_0(x0)
    x1 = cnn_layer_1(x1)
    x2 = cnn_layer_2(x2)
    x3 = cnn_layer_3(x3)
    x4 = cnn_layer_4(x4)
    x5 = cnn_layer_5(x5)
    x6 = cnn_layer_6(x6)
    x7 = cnn_layer_7(x7)
    x8 = cnn_layer_8(x8)

        
    x = tf.concat([ x0, x1, x2, x3, 
                    x4, 
                    x5, x6, 
                    x7,
                    x8,
                    ], -2)

    
    
    x = layers.BatchNormalization()(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(128, dropout=0.25, recurrent_dropout=0.25))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    answer = layers.Dense(6, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_sig, outputs=answer)

    model.summary()

    tf.keras.utils.plot_model(model, "proto_model_with_shape_info_cnn_lstm_1.png", show_shapes=True)

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

history = model.fit(train_dataset, epochs=20,
                    class_weight=model_class_weights,
                    )

#%%
eval_loss, eval_acc = model.evaluate(test_dataset)

print('\n\n')
print("loss : {}, accuracy : {}".format(eval_loss, eval_acc))
print('\n\n')



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

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['accuracy'], 'b', label='train accuracy')

acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()





