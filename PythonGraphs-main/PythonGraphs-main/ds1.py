# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:08:59 2019

@author: M GNANESHWARI


x=("Python", "version", "3.7")
print(type(x))

p=45
p=float(p)
print(type(p))


X=300 
Y= 17
X%=Y
print(X)
""""
lst=[1,2,3,4]
print(lst[-1])
print(lst*2)
"""
lst=[[1,2,3],[4,5,6]]
v=lst[0][0]
for lst in v

inventory={
        'gold':500,50
        'pouch':['flint','twine','gemstone'],
        'backpack':['xylophone','dagger','bedroll','bread loaf'],
        'pocket':['seashell','strange','berry','lint']
        }
print(inventory)

import numpy as np
a=np.linspace(start=0,stop=100,num=5,dtype=int)
print(a)

import numpy as np
b=np.linspace(start=1,stop=5,num=4,endpoint=False)
print(b)

import numpy as np
c=np.ones(shape=4,dtype=int)
print(c)

import numpy as np
a1=np.ones(shape=4)
print(a1)
a2=np.ones(4)
print(a2)
a3=np.ones(4,dtype=int)
print(a3)
a4=np.ones(shape=(2,3),dtype=int)
print(a4)

import numpy as np
a1=np.zeros(shape=4)
#print(a1)
a2=np.zeros(4)
#print(a2)
a3=np.zeros(4,dtype=int)
print(a3)
a4=np.zeros(shape=(2,3),dtype=int)
print(a4)

import numpy as np
num=np.random.normal()
print(num)
num1=np.random.normal(size=4)
print(num1)
b3=np.random.uniform(size=4)
print(b3)
b2=np.random.randint(low=1,high=100,size=4)
print(b2)

import numpy as np
a=np.logspace(1,10,num=5,base=2)
print(a)

from numpy import *
a=arange(9)
print(a)
print(a[7:-5])

from numpy import *
a=arange(27).reshape(3,3,3)
print(a)
print(a[1:3,1:3,1:3])

import numpy as np
a=np.linspace(start=0,stop=100,num=5)
print(a)

prices={"banana":4,"apple":2,"orange":1.5,"pear":3}
print(prices.keys())        
print(prices.values())
print(prices)

import numpy as np
a=np.array([[1,2],[3,4]])
b=np.array([[1,0],[0,1]])
print(a)
d=a*b
print(d)
c=a+b
print(c)
e=a-b
print(e)

import numpy as np
a=np.array([[0,2,3],[4,5,6]])
print(a)
print(a.sum())
print(a.sum(1))
print(a.sum(0))
print(a.max())
print(a.min())
print(a.mean())
print(a.var())
print(a.all())
print(a.any())

import numpy as np
a=np.array([[0,2,3],[4,5,6]])
a=np.arange(9)
b=a.reshape(3,3)
print(b)

import numpy as np
a=np.array([[0,2],[4,5]])
b=np.array([[1,0],[0,3]])
print(a)
print(b)
c=a.dot(b)
print(c)
print(a.tolist())
print(a.transpose())

import numpy as np
A=[[1,4,5,12],
[-5,8,9,0],
[-6,7,8,9]]

m=np.matrix(A)
print(m[0])
print(m[1])
print(m[2])
print(m[:,-1])
print(m[:,0])
print(m[:,2])

import numpy as np
A=np.matrix([[1,2],[2,3]])
B=np.matrix([[3,4],[5,6]])
c=np.dot(A,B)
print(c)
d=A.transpose()
e=np.dot(d,B)
print(e)
print(np.linalg.det(A))
print(np.linalg.inv(A))
print(np.linalg.matrix_rank(A))

import numpy as np
A=np.array([[1,2,3],[-3,5,4],[1,-1,5]])
b=np.transpose(np.array([[-3,5,2]]))
x=np.linalg.solve(A,b)
print(x)

plants={"raddish":2,"onion":4,"carrot":8}
print(plants.get("raddish","yes can fount"))
"""
