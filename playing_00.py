# -*- coding: utf-8 -*-

'''
self number problem
'''

max_num = 5000

temp_max = 5000

temp_list = list()

for i in range(10000):
    if i >= 1000:
        a = i//1000
        b = (i - a*1000)//100
        c = (i - a*1000 - b*100)//10
        d = (i - a*1000 - b*100 -c*10)
        e = a + b + c + d + i
        temp_list.append(e)
    if i < 1000 and i >= 100:
        a = i//100
        b = (i - a*100)//10
        c = (i - a*100 - b*10)
        d = a + b + c + i
        temp_list.append(d)
    if i < 100 and i >= 10:
        a = i//10
        b = (i - a*10)
        c = a + b + i
        temp_list.append(c)
    if i < 10:
        a = i + i
        temp_list.append(a)

answer_list = list()

for i in range(5000):
    if i not in temp_list:
        answer_list.append(i)
        
print(answer_list)
