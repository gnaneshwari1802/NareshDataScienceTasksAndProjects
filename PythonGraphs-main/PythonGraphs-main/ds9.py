# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:05:57 2019

@author: M GNANESHWARI
"""

import matplotlib.pyplot as p
import numpy as np
import pandas as pd

from math import *

x1=[1,1]
x2=[2,9]
eudistance=math.sqrt(math.pow(x1[0]-x2[0],2)+math.pow(x1[1]-x2[1],2))
print("eudistance using math",eudistance)
eudistance=spacial.distance.euclidean(x1,x2)
print("eudistance Using scipy",eudistance)
x1np=numpy.array(x1)
x2np=numpy.array(x2)
eudistance=numpy.sqrt(numpy.sum((x1np-x2np)**2))
print("eudistance using numpy",eudistance)
eudistance=numpy.linalg.norm(x1np-x2np)
print("eudistance using numpy",eudistance)
eudistance=euclidean_distance([x1np],[x2np])
print("eudistance using sklearn",eudistance)
print("***program ended***")
from math import*
def euclidean_distance(x,y):
 return sqrt(sum(pow(a-b,2) for a,b in zip(x,y)))
print(euclidean_distance([0,3,4,5],[7,6,3,-1]))
def manhattan_distance(x,y):
    return sum(abs(-b) for a,b in zip(x,y))
print("first md is",manhattan_distance([10,20,10],[10,20,20]))
print("sec md is",manhattan_distance([10,10,10],[10,20,20]))
print("third",manhattan_distance([10,20,10],[20,20,20]))

from scipy.spatial import distance
print("first md is",distance.cityblock([10,20,10],[10,20,20]))
print("sec md is",manhattan_distance([10,10,10],[10,20,20]))
print("third",manhattan_distance([10,20,10],[20,20,20]))

from scipy.spacial import distance
print("first cd is",distance.chebyshev([1,0,0],[0,1,0]))
print("first cd is",distance.chebyshev([1,1,0],[0,1,0]))
print("first cd is",distance.chebyshev([11,10,10],[20,21,20]))
print("first cd is",distance.chebyshev([11,11,10],[20,21,20]))
