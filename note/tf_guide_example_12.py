import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def print_title(title_sent):
    print('\n', title_sent)
    return

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)

print_title('tf.data 예제')
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
print(dataset)

for elem in dataset:
  print(elem.numpy())

it = iter(dataset)

print(next(it).numpy())

print(dataset.reduce(0, lambda state, value: state + value).numpy())

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))

print(dataset1.element_spec)

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

print(dataset2.element_spec)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

print(dataset3.element_spec)

# Dataset containing a sparse tensor.
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

print(dataset4.element_spec)

# Use value_type to see the type of value represented by the element spec
print(dataset4.element_spec.value_type)

dataset1 = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))

print(dataset1)

for z in dataset1:
  print(z.numpy())

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

print(dataset2)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

print(dataset3)

for a, (b,c) in dataset3:
  print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))



train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset
