# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:46:32 2019

@author: M GNANESHWARI

import numpy as np

ndarray=np.ones([3,3])
print("content of array:")
print(ndarray)
print()

print("tensorflow operations convert numpy arrays to tensors automatically")
#tensor=tf.multiply(ndarray,42)
print(tensor)
print()

print("And numpy operations convert tensors to numpy arrays automatically")

from _future_ import absolute_import,division,print_function,unicode_literals

try:
    !pip install tf-nightly-2.0-preview
except Exception:
 pass
import tensorflow as tf    

import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
dataset=tf.data.Dataset.from_tensor_slices([8,3,0,8,2,1])
print(dataset)

import numpy as np
dataset=tf.data.Dataset.from_tensor_slices([8,3,0,8,2,1])
for elen in dataset:
    print(elen.numpy())
it=iter(dataset)
print(next(it).numpy()) 
print(dataset.reduce(0,lambda state,value:[state+value).numpy())]))
"""
   