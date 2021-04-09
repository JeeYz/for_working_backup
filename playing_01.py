# -*- coding: utf-8 -*-

'''
snail output question
'''

max_num = 10

temp = list()

for i in range(max_num):
    one_row = list()
    for j in range(max_num):
        one_row.append(0)
    temp.append(one_row)

i,j  = 0,0

for val in range(1, max_num**2+1):
    if i == 0:
        pass
    elif j == 0:
        pass
    elif i == 9:
        pass
    elif j == 9:
        pass
    else:
        pass        


for i in temp:
    print(i)




