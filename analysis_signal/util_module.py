import numpy as np

##
def standardization_func(data):
    return (data-np.mean(data))/np.std(data)


#%% new min max normal
def new_minmax_normal(input_data):
    temp = list()
    for one in input_data:
        one = np.array(one, dtype='float32')
        res = (one - np.min(one))/(np.max(one) - np.min(one))
        # res = res*2 - 1.0
        temp.append(res)

    res_data = np.array(temp)

    return res_data


##
def transpose_the_matrix(data):
    return np.swapaxes(data, 0, 1)







## endl
