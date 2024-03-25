# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:02:20 2019

@author: M GNANESHWARI
"""

import numpy as np
b=np.array([1,2,3])
print(b)
f=np.array([2,3,4,5],dtype=float)
print(f)
a=np.arange(10)
print(a)
e=np.array([np.arange(4),np.arange(4)])
print(e)
c=np.arange(9)
d=c.reshape([3,3])
print(d)

mytype=np.dtype([('rno',np.int32),('name',np.str_,30),('ml',np.int32),('m2',np.int32),('m3',np.int32)])
print(mytype)
student=[(1,'rama',45,54,45),(2,'krishna',54,43,43)]
std=np.array(student,dtype=mytype)
print(std)
                                                   