#%%
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import copy
import json

from tensorflow.python.keras.backend import conv2d, dropout

CONVERT_TFLITE_BOOL = True
RESULT_GRAPH_BOOL = True

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'

traindata_json_file = 'C:\\temp\\npz_data.json'
# train_data_path = "D:\\GEN_train_data_Ver.1.0_CWdata.npz"

tflite_file_path = "PNC_ASR_2.4_CW_model_.tflite"
saved_model_file = "C:\\temp\\experiment_model_"


#%% variables
# 에포크 
EPOCH_NUM = 10
# CNN 초기값
INIT_CNN_CHAN = 16

# AE 초기값
INIT_AE_CHAN = 4

# LSTM 셀 초기값
INIT_LSTM_CELL = 128

# ResNet 채널 초기값
INIT_RESNET_CH = 32

# 뉴런의 개수
TAIL_DENSE_NUM = 128
NORM_DENSE_NUM = 128

# 신호 나누는 값 기준
DIVIDE_SIZE = 4000
DIV_SHIFT_SIZE = 2000
FULL_SIZE = 40000

# 모델 레이어 값
DENSE_LAYERS_NUM = 3
CNN_LAYERS_NUM = 7
LSTM_LAYERS_NUM = 3
BILSTM_LAYERS_NUM = 3
RESNET_LAYERS_NUM = 5
TAIL_DENSE_LAYERS_NUM = 2

# 전처리 부분 레이어 값
DENSE_FEATURE_NUM = 3
CONV_FEATURE_NUM = 5
RESNET_FEATURE_NUM = 4

# 배치 사이즈
TRAIN_BATCH_SIZE = 128
# 레이블 개수
NUM_LABELS = 32

RESIZE_BOOL = True

RESIZE_X = 64
RESIZE_Y = 64




#%%
# loaded_data_00 = np.load(train_data_path, allow_pickle=True)
# loaded_data_00 = np.load(train_data_path)
# train_data_00 = loaded_data_00['data']
# train_label_00 = loaded_data_00['label']

# loaded_data_01 = np.load(test_data_path, allow_pickle=True)
# loaded_data_01 = np.load(test_data_path)
# test_data_00 = loaded_data_01['data']
# test_label_00 = loaded_data_01['label']

with open(traindata_json_file, 'r', encoding='utf-8') as fr:
    loaded_data = json.load(fr)




#%% class -> residual cnn block
class residual_block(layers.Layer):

    def __init__(self, ch):
        super(residual_block, self).__init__()
        self.ch_size = ch

        residual_block_layer = list()

        conv2d_layer_1 = layers.Conv2D(self.ch_size, (3, 3), padding='same')
        conv2d_layer_2 = layers.Conv2D(self.ch_size, (3, 3), padding='same')

        self.conv1 = layers.Conv2D(self.ch_size, (1, 1), padding='same')

        self.res_block_layer_A = tf.keras.Sequential()

        self.res_block_layer_A.add(conv2d_layer_1)
        self.res_block_layer_A.add(layers.BatchNormalization())
        self.res_block_layer_A.add(tf.keras.layers.ReLU(max_value=6.0))

        self.res_block_layer_A.add(conv2d_layer_2)
        self.res_block_layer_A.add(layers.BatchNormalization())


    def __call__(self, input_x):

        # x = self.residual_block_layer(input_x)

        x = self.res_block_layer_A(input_x)
        y = self.conv1(input_x)

        x = tf.math.add(y, x)
        x = tf.nn.relu6(x)
        x = layers.Dropout(0.2)(x)

        return x



#%%
class residual_layers(layers.Layer):
    def __init__(self, num_of_layers):
        super(residual_layers, self).__init__()

        init_ch = INIT_RESNET_CH
        # self.init_ch = INIT_RESNET_CH
        
        self.layers_list = list()

        self.res_layers_list = list()

        for i in range(1, (num_of_layers+1)):
            temp = residual_block(init_ch*i)
            self.res_layers_list.append(temp)
            
            temp = layers.Conv2D(init_ch*i, (3, 3), activation=None, padding='same', strides=2)
            self.res_layers_list.append(temp)
        
        self.res_layers_list.append(layers.Flatten())


    def __call__(self, input_x):

        x = input_x

        for one_layer in self.res_layers_list:
            x = one_layer(x)
            
        return x



#%%
class autoencoder_cnn_block(layers.Layer):
    def __init__(self, **kwargs):
        super(autoencoder_cnn_block, self).__init__()
        
        init_ch = INIT_AE_CHAN
        
        self.encoder = tf.keras.Sequential([ 
                    layers.Conv2D(init_ch, (3,3), activation='relu', padding='same', strides=2),
                    layers.Conv2D(init_ch*2, (3,3), activation='relu', padding='same', strides=2),
                    layers.Conv2D(init_ch*4, (3,3), activation='relu', padding='same', strides=2),])

        self.decoder = tf.keras.Sequential([
                    layers.Conv2DTranspose(init_ch*4, kernel_size=3, strides=2, activation='relu', padding='same'),
                    layers.Conv2DTranspose(init_ch*2, kernel_size=3, strides=2, activation='relu', padding='same'),
                    layers.Conv2DTranspose(init_ch, kernel_size=3, strides=2, activation='relu', padding='same'),
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



#%%
class dense_layers(layers.Layer):
    def __init__(self, num_of_layers, **kwargs):
        super(dense_layers, self).__init__()

        if "dropout_bool" in kwargs.keys():
            dropout_bool = kwargs["dropout_bool"]
        else:
            dropout_bool = True

        layers_list = list()

        layers_list.append(layers.BatchNormalization())
        for i in reversed(range(1, num_of_layers+1)):
            layers_list.append(layers.Dense(NORM_DENSE_NUM*i))
            if dropout_bool == True:
                layers_list.append(layers.Dropout(0.25))

        self.dlayers = tf.keras.Sequential(layers_list)

    def __call__(self, input_x):
        return self.dlayers(input_x)



#%%
class conv2d_layers(layers.Layer):
    def __init__(self, num_of_layers):
        super(conv2d_layers, self).__init__()

        init_ch_num = INIT_CNN_CHAN
        layers_list = list()

        layers_list.append(layers.BatchNormalization())
        for i in range(num_of_layers):
            layers_list.append(layers.Conv2D(   
                                                init_ch_num*(i+1), 
                                                (3, 3), 
                                                padding='same', 
                                                activation='relu', 
                                                strides=2,
                                            ))
            layers_list.append(layers.BatchNormalization())
            
        layers_list.append(layers.Flatten())
        layers_list.append(layers.Dropout(0.25))

        self.clayers = tf.keras.Sequential(layers_list)        

    def __call__(self, input_x):
        return self.clayers(input_x)



#%%
class lstm_layers(layers.Layer):
    def __init__(self, num_of_layers):
        super(lstm_layers, self).__init__()

        layers_list = list()
        init_cells = INIT_LSTM_CELL

        layers_list.append(layers.BatchNormalization())
        for i in range(num_of_layers):
            layers_list.append(layers.LSTM(init_cells, return_sequences=True))

        layers_list.append(layers.LSTM(init_cells))

        self.lslayers = tf.keras.Sequential(layers_list)

    def __call__(self, input_x):
        return self.lslayers(input_x)



#%%
class bilstm_layers(layers.Layer):
    def __init__(self, num_of_layers):
        super(bilstm_layers, self).__init__()

        layers_list = list()
        init_cells = INIT_LSTM_CELL

        layers_list.append(layers.BatchNormalization())
        for i in range(num_of_layers):
            layers_list.append(layers.Bidirectional(layers.LSTM(init_cells, return_sequences=True)))
        
        layers_list.append(layers.Bidirectional(layers.LSTM(init_cells)))

        self.bilslayers = tf.keras.Sequential(layers_list)

    def __call__(self, input_x):
        return self.bilslayers(input_x)


class tail_dense_layers(layers.Layer):
    def __init__(self, num_of_layers, **kwargs):
        super(tail_dense_layers, self).__init__()

        if "dropout_bool" in kwargs.keys():
            dropout_bool = kwargs["dropout_bool"]
        else:
            dropout_bool = True

        layers_list = list()

        # layers_list.append(layers.Flatten())
        layers_list.append(layers.BatchNormalization())
        for i in range(num_of_layers):
            layers_list.append(layers.Dense(TAIL_DENSE_NUM))
            if dropout_bool == True:
                layers_list.append(layers.Dropout(0.25))

        self.dlayers = tf.keras.Sequential(layers_list)

    def __call__(self, input_x):
        return self.dlayers(input_x)
    
    
    
#%%
class preprocessing_layer(layers.Layer):
    def __init__(self, **kwargs):
        super(preprocessing_layer, self).__init__()
        self.divided_num = self.make_div_number()
        
        if "prepro_flag" in kwargs.keys():
            self.prepro_flag = kwargs["prepro_flag"]
        else:
            self.prepro_flag = None
            
        if "resize_bool" in kwargs.keys():
            self.resize_bool = kwargs["resize_bool"]
        else:
            self.resize_bool = False
        
        self.pre_layers = list()
        
        for i in range(self.divided_num):
            temp = self.gen_layers()
            self.pre_layers.append(temp)
            
        
    
    def make_div_number(self):
        
        return (FULL_SIZE-DIVIDE_SIZE)//DIV_SHIFT_SIZE+1
    
    
    def gen_layers(self):
        
        if self.prepro_flag == 'cnn':
            temp_layer = conv2d_layers(CONV_FEATURE_NUM)
    
        elif self.prepro_flag == 'resnet':
            temp_layer = residual_layers(RESNET_FEATURE_NUM)
            
        elif self.prepro_flag == 'ae_dense':
            temp_layer = autoencoder_snn_block()
    
        elif self.prepro_flag == 'ae_cnn':
            temp_layer = autoencoder_cnn_block()
    
        else:
            temp_layer = dense_layers(DENSE_FEATURE_NUM)
        
        return temp_layer
        
    
    
    def stft_function(self, input_x):

        x = tf.signal.stft(input_x, frame_length=255, frame_step=128)
        x = tf.abs(x)
        x = tf.expand_dims(x, -1)

        if self.resize_bool == True:    
            x = preprocessing.Resizing(RESIZE_X, RESIZE_Y)(x)
            
        x = layers.BatchNormalization()(x)

        return x    
    
    
    def concat_function_2(self, input_x):

        x = tf.concat(input_x, -2)

        return x
    
        
    def __call__(self, input_x):
        
        new_data_list = list()
        
        for one_d, one_layer in zip(input_x, self.pre_layers):
            if self.prepro_flag == 'ae_cnn' or self.prepro_flag == 'cnn' or self.prepro_flag == 'resnet':
                one_d = self.stft_function(one_d)
            temp_data = one_layer(one_d)
            temp_data = layers.Flatten()(temp_data)
            temp_data = tf.expand_dims(temp_data, -2)
            new_data_list.append(temp_data)
            
        x = self.concat_function_2(new_data_list)        
        return x
        
    
         


#%% 실험 1-1
class experiment_models(layers.Layer):
    def __init__(self, **kwargs):
        super(experiment_models, self).__init__()


        if "num_of_labels" in kwargs.keys():
            self.num_of_labels = kwargs["num_of_labels"]
        else:
            self.num_of_labels = NUM_LABELS

        self.dense_block = dense_layers(DENSE_LAYERS_NUM)
        self.conv2d_block = conv2d_layers(CNN_LAYERS_NUM)
        self.lstm_block = lstm_layers(LSTM_LAYERS_NUM)
        self.bilstm_block = bilstm_layers(BILSTM_LAYERS_NUM)
        self.residual_blocks = residual_layers(RESNET_LAYERS_NUM)
        self.autoencoder_cnn_blocks = autoencoder_cnn_block()
        self.autoencoder_nn_blocks = autoencoder_snn_block()
        self.tail_block = tail_dense_layers(TAIL_DENSE_LAYERS_NUM)
        
        self.pre_proc_layer = preprocessing_layer(prepro_flag = 'resnet',
                                                    resize_bool = RESIZE_BOOL,
                                                  )

    
    def divide_function(self, input_x, **kwargs):

        full_size = FULL_SIZE

        dv_size = DIVIDE_SIZE
        sh_size = DIV_SHIFT_SIZE

        dv_position = 0

        data_list = list()
        
        while True:
            
            temp = tf.slice(input_x, begin=[0, dv_position], size=[-1, dv_size])
            data_list.append(temp)
            dv_position = dv_position+sh_size

            if (dv_position+dv_size) > full_size:
                break
            
        return data_list

        

    def __call__(self, input_x):
        
        
        # RNN 계열 모델
        
        # x = self.divide_function(input_x)
        # x = self.pre_proc_layer(x)
        # # 
        # # x = self.lstm_block(x)
        # x = self.bilstm_block(x)
        
        
        # CNN 계열 모델

        # x = self.autoencoder_nn_blocks(x)
        # x = self.autoencoder_cnn_blocks(x)
        
        # x = self.stft_function(input_x)
        x = self.pre_proc_layer.stft_function(input_x)
        
        # x = self.conv2d_block(x)
        x = self.residual_blocks(x)
        
        # x = self.dense_block(x)
        
        x = self.tail_block(x)
    

        result = layers.Dense(self.num_of_labels, activation='softmax')(x)

        return result


#%%
mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


#%%
with mirrored_strategy.scope():

    input_sig = tf.keras.Input(shape=(FULL_SIZE,))

    experiment_model_layers = experiment_models()
    
    answer = experiment_model_layers(input_sig)

    model = tf.keras.Model(inputs=input_sig, outputs=answer)

    model.summary()

    tf.keras.utils.plot_model(model, "proto_model_with_shape_info_cnn_lstm_1.png", show_shapes=True)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

# def generate_train_data(train_data, train_label):
#     for one_data, one_label in zip(train_data, train_label):
#         yield (one_data, one_label)

# @tf.function
def generate_train_data():
    for idx in range(len(loaded_data)):
        npz_file = loaded_data[idx]
        loaded_numpy = np.load(npz_file)
        train_data = loaded_numpy['data']
        train_label = loaded_numpy['label']

        for one_data, one_label in zip(train_data, train_label):
            yield (one_data, one_label)

# def generate_test_data():
#     for one_data, one_label in zip(test_data_00, test_label_00):
#         yield (one_data, one_label)


#%%
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

#%%
train_dataset = tf.data.Dataset.from_generator(
    generate_train_data, 
    # output_types=(tf.float32, tf.int32),
    output_signature=(
        tf.TensorSpec(shape=(FULL_SIZE, ), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ),
).shuffle(5000).batch(TRAIN_BATCH_SIZE)
# args=(train_data_path)),
train_dataset = train_dataset.with_options(options)
train_dataset = train_dataset.cache()


#%%
history = model.fit(
    train_dataset, 
    epochs=EPOCH_NUM,
)


#%%
# eval_loss, eval_acc = model.evaluate(test_dataset)

# print('\n\n')
# print("loss : {}, accuracy : {}".format(eval_loss, eval_acc))
# print('\n\n')





#%% F1 Score 계산하기
# labels_num = NUM_LABELS

# confusion_matrix = np.zeros((labels_num, labels_num), dtype=np.int16)

# length_of_true_ans = len(test_label_00)

# for i in range(length_of_true_ans):
#     x_axis = int(test_label_00[i])
#     y_axis = int(result_arg[i])
#     confusion_matrix[x_axis, y_axis]+=1


# print('\n')
# print(confusion_matrix)
# print('\n')

# sum_ax_0 = np.sum(confusion_matrix, axis=0)
# sum_ax_1 = np.sum(confusion_matrix, axis=1)

# print('\n')
# print(sum_ax_0)
# print(sum_ax_1)
# print('\n')

# precisions = list()
# recalls = list()

# for i in range(labels_num):

#     if sum_ax_0[i] != 0:
#         temp_val = confusion_matrix[i][i]/sum_ax_0[i]
#         precisions.append(temp_val)
#     else:
#         precisions.append(0.0)

#     if sum_ax_1[i] != 0:
#         temp_val = confusion_matrix[i][i]/sum_ax_1[i]
#         recalls.append(temp_val)
#     else:
#         precisions.append(0.0)

# f_result = open("D:\\result_acc.txt", "a")

# print('\n')
# f_result.write("\n\n")
# for i in range(labels_num):
#     f1_score = 2*(precisions[i]*recalls[i])/(precisions[i]+recalls[i])
#     print("{} label : precision : {},    recall : {},    f1 score : {}".format(i, precisions[i], recalls[i], f1_score))
#     f_result.write("{} label : precision : {},    recall : {},    f1 score : {}".format(i, precisions[i], recalls[i], f1_score))
#     f_result.write("\n")

# print('\n')
# f_result.write("\n")

# temp_sum = 0
# for i in range(labels_num):
#     temp_sum+=confusion_matrix[i][i]

# total_acc = temp_sum/np.sum(sum_ax_0)
# total_acc_1 = temp_sum/len(test_label_00)

# print("number of test data : {a}".format(a=len(test_label_00)))
# f_result.write("number of test data : {a}".format(a=len(test_label_00)))
# f_result.write("\n")
# print("sum of right answers : {b}".format(b=temp_sum))
# f_result.write("sum of right answers : {b}".format(b=temp_sum))
# f_result.write("\n")
# print("sum of sum_ax_0 : {}".format(np.sum(sum_ax_0)))
# f_result.write("sum of sum_ax_0 : {}".format(np.sum(sum_ax_0)))
# f_result.write("\n")
# print("sum of sum_ax_1 : {}".format(np.sum(sum_ax_1)))
# f_result.write("sum of sum_ax_1 : {}".format(np.sum(sum_ax_1)))
# f_result.write("\n")
# print("accuracy : {}".format(total_acc))
# f_result.write("accuracy : {}".format(total_acc))
# f_result.write("\n")
# print("accuracy 1 : {}".format(total_acc_1))
# f_result.write("accuracy 1 : {}".format(total_acc_1))
# f_result.write("\n")

# f_result.close()




#%% 훈련 데이터 테스트 F1 Score 계산하기
# labels_num = NUM_LABELS

# confusion_matrix = np.zeros((labels_num, labels_num), dtype=np.int64)

# # print('\n')
# # print(confusion_matrix)
# # print('\n')

# length_of_true_ans = len(train_data_00)

# for i in range(length_of_true_ans):
#     x_axis = int(train_label_00[i])
#     y_axis = int(result_arg_train_data[i])
#     confusion_matrix[x_axis, y_axis]+=1


# print('\n')
# print(confusion_matrix)
# print('\n')

# sum_ax_0 = np.sum(confusion_matrix, axis=0)
# sum_ax_1 = np.sum(confusion_matrix, axis=1)

# print('\n')
# print(sum_ax_0)
# print(sum_ax_1)
# print('\n')

# precisions = list()
# recalls = list()

# for i in range(labels_num):

#     if sum_ax_0[i] != 0:
#         temp_val = confusion_matrix[i][i]/sum_ax_0[i]
#         precisions.append(temp_val)
#     else:
#         precisions.append(0.0)

#     if sum_ax_1[i] != 0:
#         temp_val = confusion_matrix[i][i]/sum_ax_1[i]
#         recalls.append(temp_val)
#     else:
#         precisions.append(0.0)

# print('\n')

# for i in range(labels_num):
#     f1_score = 2*(precisions[i]*recalls[i])/(precisions[i]+recalls[i])
#     print("{} label : precision : {},    recall : {},    f1 score : {}".format(i, precisions[i], recalls[i], f1_score))
    
# print("\n\n")
    
# temp_sum = 0
# for i in range(labels_num):
#     temp_sum+=confusion_matrix[i][i]

# total_acc = temp_sum/np.sum(sum_ax_0)
# total_acc_1 = temp_sum/len(train_label_00)

# print("number of test data : {a}".format(a=len(train_label_00)))

# print("sum of right answers : {b}".format(b=temp_sum))

# print("sum of sum_ax_0 : {}".format(np.sum(sum_ax_0)))

# print("sum of sum_ax_1 : {}".format(np.sum(sum_ax_1)))

# print("accuracy : {}".format(total_acc))

# print("accuracy 1 : {}".format(total_acc_1))






#%%
# convert tflite

if CONVERT_TFLITE_BOOL == True:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                            tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(tflite_file_path, 'wb').write(tflite_model)




#%%
if RESULT_GRAPH_BOOL == True:
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





