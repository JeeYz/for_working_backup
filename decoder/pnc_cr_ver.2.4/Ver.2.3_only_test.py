
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

tflite_file = "D:\\PNC_ASR_2.3_GEN_train_data_1.0_.tflite"
testdata_file = "D:\\test_data_20000_.npz"
h5_file_for = "D:\\Ver.2.3_model.h5"


# 레이블 개수
NUM_LABELS = 6
TRAIN_BATCH_SIZE = 256
FULL_SIZE = 20000


#%%
loaded = np.load(testdata_file)
test_data = loaded['data']
test_label = loaded['label']



#%%
mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


#%%
model = tf.keras.models.load_model(h5_file_for)


#%%
check_list = list()

def generate_test_data():

    for one_data, one_label in zip(test_data, test_label):
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
eval_loss, eval_acc = model.evaluate(test_dataset)

print('\n')
print("loss : {}, accuracy : {}".format(eval_loss, eval_acc))
print('\n')

result_prediction = model.predict(test_dataset)
#%%
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





