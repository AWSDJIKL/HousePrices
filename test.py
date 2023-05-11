# -*- coding: utf-8 -*-
'''

'''
# @Time : 2023/5/8 11:22
# @Author : LINYANZHEN
# @File : test.py
import numpy as np

a=np.array([1,2,3,4])
b=np.array([2,3,4,5])
print(np.sqrt(np.mean(np.sum(np.power((a - b), 2)))))