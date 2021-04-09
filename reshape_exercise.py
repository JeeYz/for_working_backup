# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import MinMaxScaler

a = np.random.randint(-10, 10, size=(5, 5, 3))
a = np.asarray(a, dtype='float64')

# print(a)

b = np.reshape(a, (5, -1))

print(b)
print(b.shape)

print(np.min(b))
print(np.max(b))

scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(b)
c = scaler.transform(b)
print(c)

d = (b - np.min(b))/(np.max(b) - np.min(b))
print(b)

e = d*2-1.0
print(e)

f = np.apply_along_axis(lambda a: np.min(a), 1, b)
print(f)
