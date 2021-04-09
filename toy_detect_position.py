import numpy as np


a0 = list()
a1 = list()
a2 = list()
distance_r = list()

def detect_position(**kwarg):

    if "anchor_0" in kwarg.keys():
        a0 = kwarg['anchor_0']
    if "anchor_1" in kwarg.keys():
        a1 = kwarg['anchor_1']
    if "anchor_2" in kwarg.keys():
        a2 = kwarg['anchor_2']
    if "distance_list" in kwarg.keys():
        distance_r = kwarg['distance_list']

    x = ((np.power(distance_r[0], 2)-np.power(distance_r[2], 2))/a2[0]+a2[0])*0.5
    y = ((np.power(distance_r[0], 2)-np.power(distance_r[1], 2))/a1[1]+a1[1])*0.5
    z = -1*np.sqrt((np.power(distance_r[0], 2)-np.power(x, 2)-np.power(y, 2)))

    return x, y, z
