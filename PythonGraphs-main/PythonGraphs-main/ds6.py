
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:30:29 2019

@author: M GNANESHWARI
"""
"""
print("\n"*9)
print()
print("\t")

"""
"""
import pandas as pd
n = int(input("Enter the number of rows in a matrix: "))
a = [[0] * n for i in range(n)]
col_names = []
row_names = []


for i in range(n):
    col_names.append('col ' + str(i+1))
    row_names.append('row ' + str(i+1))  
    for j in range(n):
          a[i][j] = int(input())

print(pd.DataFrame(a,columns = col_names, index = row_names))
"""
"""
import pandas as pd
n = int(input("Enter the number of rows in a matrix: "))
a = [[0] * n for i in range(n)]
col_names = []
row_names = []


for i in range(n):
    col_names.append('col ' + str(i+1)).
    row_names.append('row ' + str(i+1))  
    for j in range(n):
          a[i][j] = int(input())

print(pd.DataFrame(a,columns = col_names, index = row_names))

"""
# A Naive recursive Python implementation of LCS problem 
"""  
def lcs(X, Y, m, n): 
  
    if m == 0 or n == 0: 
       return 0; 
    elif X[m-1] == Y[n-1]: 
       return 1 + lcs(X, Y, m-1, n-1); 
    else: 
       return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n)); 
  
  
# Driver program to test the above function 
X = "AGGTAB"
Y = "GXTXAYB"
print "Length of LCS is ", lcs(X , Y, len(X), len(Y)) 
"""
"""
a =  "codementor"
>>> print "Reverse is",a[::-1]
Reverse is rotnemedoc
"""
"""
mat = [[1, 2, 3], [4, 5, 6]]
>>> zip(*mat)
[(1, 4), (2, 5), (3, 6)]
"""
"""
>>> a = [1, 2, 3]
>>> x, y, z = a 
>>> x
1
>>> y
2
>>> z
3
"""
"""
>>> print " ".join(a)
Code mentor Python Developer
"""
"""
>>> for x, y in zip(list1,list2):
...    print x, y
...
a p
b q
c r
d s
"""
"""
 a=7
>>> b=5
>>> b, a =a, b
>>> a
5
>>> b
7
"""
"""
>>> print "code"*4+' '+"mentor"*5
codecodecodecode mentormentormentormentormentor
"""
"""
>>> import itertools 
>>> list(itertools.chain.from_iterable(a))
[1, 2, 3, 4, 5, 6]
"""
"""
def is_anagram(word1, word2):
    """Checks whether the words are anagrams.
    word1: string
    word2: string
    returns: boolean
    """
    """
    """
    from collections import Counter
def is_anagram(str1, str2):
     return Counter(str1) == Counter(str2)
>>> is_anagram('abcd','dbca')
True
>>> is_anagram('abcd','dbaa')
False
"""
"""
>>> result = map(lambda x:int(x) ,raw_input().split())
1 2 3 4
>>> result
[1, 2, 3, 4]
"""