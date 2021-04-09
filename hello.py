# -*- coding: utf-8 -*-


print("hello~world")
      
      
print("hello~, DL world~!")


d = dict()

d = {1:2, 3:4, 5:6}

print(d)

for key, value in d.items():
    print(key, value)
    
def print_dict(**kwarg):
    print(kwarg)
    
    if 'para' in kwarg.keys():
        if kwarg['para'] in d.keys():
            print('There is the key \"%d\" in that dictionary'\
                  %(kwarg['para']))
            print('There is the key \"{number}\" in that dictionary'\
                  .format(number = kwarg['para']))
        else:
            print("Sorry, There is not in that dictionary")
    else:
        print("Sorry~!!")


print_dict(para = 1)

a = (1, 2, 3)
b = (4, 5)
c = (6, )

print(a+b+c)


    