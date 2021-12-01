#%%
from numpy.core.numeric import full
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.eager.context import check_alive
from sklearn.metrics import f1_score, precision_recall_fscore_support

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

import os
import copy
import json
import random

import gen_testdata_process as gentest
from global_variables import *

from tensorflow.python.keras.backend import conv2d, dropout

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'

test_data_npz_file = "C:\\temp\\test_Ver.2.4.npz"
model_saved_file = "C:\\temp\\Ver.2.4_model.h5"

traindata_json_file = 'C:\\temp\\$$npz_data.json'

# 레이블 개수
NUM_LABELS = 32
TRAIN_BATCH_SIZE = 256
FULL_SIZE = 40000

#%%
speaker_exception = {
    'strong':[
        '59185653',
        '59184297',
        '59184290',
        '59185228',
    ],
    'weak':[
        '59183640',
        '59184387',
        '59185145',
        '59185238',
        '59185713',
        '59184211',
        '59192927',
    ],
    'machine_noise':[
        '59184088',
        '59185312',
        '59185353',
        '59185622',
        '59185031',
        '59184205',
        '59184752',
    ],
    'none':[
        '59185990',
        '59198984',
    ],
}


#%%
cmd_F_num = 15
cmd_M_num = 15

def choose_speakers(input_data):
    cmd_list = list()
    noncmd_list = list()

    for one_file in input_data:
        if 'noncmd' in one_file:
            noncmd_list.append(one_file)

    print('None Label data : {num}'.format(num=len(noncmd_list)))    
    
    for one_file in input_data:
        if one_file not in noncmd_list:
            cmd_list.append(one_file)
            
    return_list = list()
    temp_num = len(noncmd_list)
    return_list = noncmd_list

    f_num = 0
    m_num = 0
    for one_file in cmd_list:
        if "cmd_F" in one_file:
            if f_num >= cmd_F_num:
                continue
            return_list.append(one_file) 
            f_num+=1
        else:
            if m_num >= cmd_M_num:
                continue
            return_list.append(one_file) 
            m_num+=1

    print('Label data : {num}'.format(
        num=len(return_list)-temp_num))    

    return return_list


def select_traindata(loaded_data):
    target_list = list()
    temp_list = list()

    speaker_keys = list(speaker_exception.keys())
    for one_key in speaker_keys:
        for one_speaker in speaker_exception[one_key]:
            temp_list.append(one_speaker)

    # temp_list -> list of speakers
    for one_speaker in temp_list:
        for one_file in loaded_data:
            if one_speaker in one_file:
                target_list.append(one_file)

    for one_file in target_list:
        loaded_data.remove(one_file)

    return loaded_data


test_F_num = 3
test_M_num = 3
    
def choose_test_data(full_list, train_list):
    temp = list()

    for one_file in full_list:
        if one_file not in train_list:
            temp.append(one_file)

    print(len(temp))

    result = list()

    num_F = 0
    num_M = 0

    remain_list = list()

    for one in temp:
        # if num_F >= test_F_num and num_M >= test_M_num:
        #     break
        if '_cmd_F' in one and num_F < test_F_num:
            result.append(one)
            num_F += 1
        elif '_cmd_M' in one and num_M < test_M_num:
            result.append(one)
            num_M += 1
        else:
            remain_list.append(one)
        

    return result, remain_list


with open(traindata_json_file, 'r', encoding='utf-8') as fr:
    loaded_data = json.load(fr)

import time
print(len(loaded_data))
select_traindata(loaded_data)
print(len(loaded_data))

return_data = choose_speakers(loaded_data)
print(len(return_data))
test_data_list, remain_list = choose_test_data(loaded_data, return_data)
print(len(test_data_list), len(remain_list))

# test_data_list = gentest.gen_testdata_for_cw(test_data_list)

test_data_list = fpro.find_data_files(
    "C:\\temp\\CW_test\\",
    '.npz',
)


#%%
mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


#%%
model = tf.keras.models.load_model(model_saved_file)


#%%
check_list = list()

def generate_test_data():
    for npz_file in test_data_list:
        loaded_numpy = np.load(npz_file)
        train_data = loaded_numpy['data']
        train_label = loaded_numpy['label']
        check_list.append(train_label)

        for one_data, one_label in zip(train_data, train_label):
            yield (one_data, one_label)


#%%
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

#%%
test_dataset = tf.data.Dataset.from_generator(
    generate_test_data, 
    # output_types=(tf.float32, tf.int32),
    output_signature=(
        tf.TensorSpec(shape=(FULL_SIZE, ), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ),
).batch(TRAIN_BATCH_SIZE)
test_dataset = test_dataset.with_options(options)


#%%
def return_labels(input_list):
    result = list()

    for one_file in input_list:
        print("**", one_file)
        loaded = np.load(one_file)
        labels = loaded['label']
        print(len(labels))
        for one_label in labels:
            result.append(one_label)
    
    return result

test_label = return_labels(test_data_list)


#%%
eval_loss, eval_acc = model.evaluate(test_dataset)

print('\n')
print("loss : {}, accuracy : {}".format(eval_loss, eval_acc))
print('\n')

result_prediction = model.predict(test_dataset)
print(result_prediction.shape)
print("Input labels size : {num}".format(num=len(test_label)))

result_arg = np.argmax(result_prediction, axis=-1)

print('\n')
print("prediction size : {}".format(len(result_arg)))   
print('\n')


#%% F1 Score 계산하기
labels_num = NUM_LABELS

confusion_matrix = np.zeros((labels_num, labels_num), dtype=np.int16)

length_of_true_ans = len(test_label)

for i in range(length_of_true_ans):
    x_axis = int(test_label[i])
    y_axis = int(result_arg[i])
    confusion_matrix[x_axis, y_axis]+=1


print('\n')
print(confusion_matrix)
print('\n')

f1_score_of_total = f1_score(test_label, result_arg, average='weighted')
print("** f1 score : {f1_score}".format(f1_score=f1_score_of_total))
pre_rec_f1_total = precision_recall_fscore_support(test_label, result_arg, average='weighted')
print("** precision : {pre}".format(pre=pre_rec_f1_total[0]))
print("** recall : {rec}".format(rec=pre_rec_f1_total[1]))


sum_ax_0 = np.sum(confusion_matrix, axis=0)
sum_ax_1 = np.sum(confusion_matrix, axis=1)

right_ans = list()
for i,m in enumerate(confusion_matrix):
    for j,n in enumerate(m):
        if i==j:
            right_ans.append(n)

print('\n')
print(sum_ax_0)
print(sum_ax_1)
print(right_ans)
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
        recalls.append(0.0)

f_result = open("D:\\result_acc.txt", "a")

print('\n')
f_result.write("\n\n")
for i in range(labels_num):
    if (precisions[i]+recalls[i]) != 0:
        f1_score = 2*(precisions[i]*recalls[i])/(precisions[i]+recalls[i])
    else:
        f1_score = 0.0
    print("{} label : precision : {},    recall : {},    f1 score : {}".format(i, precisions[i], recalls[i], f1_score))
    f_result.write("{} label : precision : {},    recall : {},    f1 score : {}".format(i, precisions[i], recalls[i], f1_score))
    f_result.write("\n")

print('\n')
f_result.write("\n")

temp_sum = 0
for i in range(labels_num):
    temp_sum+=confusion_matrix[i][i]

if np.sum(sum_ax_0) != 0:
    total_acc = temp_sum/np.sum(sum_ax_0)
    total_acc_1 = temp_sum/len(test_label)
else:
    total_acc = 0
    total_acc_1 = temp_sum/len(test_label)

print("number of test data : {a}".format(a=len(test_label)))
f_result.write("number of test data : {a}".format(a=len(test_label)))
f_result.write("\n")
print("sum of right answers : {b}".format(b=temp_sum))
f_result.write("sum of right answers : {b}".format(b=temp_sum))
f_result.write("\n")
print("sum of sum_ax_0 : {}".format(np.sum(sum_ax_0)))
f_result.write("sum of sum_ax_0 : {}".format(np.sum(sum_ax_0)))
f_result.write("\n")
print("sum of sum_ax_1 : {}".format(np.sum(sum_ax_1)))
f_result.write("sum of sum_ax_1 : {}".format(np.sum(sum_ax_1)))
f_result.write("\n")
print("accuracy : {}".format(total_acc))
f_result.write("accuracy : {}".format(total_acc))
f_result.write("\n")
print("accuracy 1 : {}".format(total_acc_1))
f_result.write("accuracy 1 : {}".format(total_acc_1))
f_result.write("\n")

f_result.close()





