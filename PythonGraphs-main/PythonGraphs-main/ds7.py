# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 21:09:03 2019

@author: M GNANESHWARI


def gcd(m,n):
    m=int(input())
    
    fm=[]
    
    for i in range(1,m+1):
        if(m%i) ==0:
            fm.append(i)
            print(fm)  

list=[]
list.append(1)
list.append(2)
list.append(3)
list.append(5.6)
print(list.sort())
list.append("Ram") 
list.append('A')           
print(list,end=" ")
print(list[3])
print(list[5])
#print(list[6])

a=[1,2,3]
b=[4,5,6]
a.extend(b)
print(a)
print(len(a))

it=["computer",1,1.5,"mark","atlas"]
'''if "computer" in it:
    print(1)'''
if "me" in it:
  print(1)    
else:
  print(0) 

list=[100,500,200,400] 
list.reverse()
print(list) 
list.sort()
print(list)
list.remove(500)
del list[2:]
del list[:1]
print(list)

list=['Tommy','bill','janet','stacy','a','a']
print(list)
f=list.count('a')
print(f)
n=list.index("a")
print(n)

values=[-11,1,10,12]
print(min(values))
"""
'''
n=int(input())
m=int(input())
name=[]
mark=[]
for i in range(1,n+1):
    names=str(input())
    name.append(names)
  for j in range(1,m+1):
        marks=int(input())
        mark.append(marks)
    print(name[i])
   print(mark[j])
 
names=[]
avg1=[]
a=int(input("no of students:"))
b=int(input("no of subjects:"))
for i in range(1,a+1):
    name=str(input("enter name of subject %d:"%i))
    names.append(name)
    marks=[]
    for j in range(1,b+1):
        mark=int(input("enter marks for that subject %d:"%j))
        marks.append(mark)
    tot=sum(int(j) for j in marks)
    print("student",i,"marks are:",marks)
    print("total marks are")
    print(marks)
    avg=tot/b
    avg1.append(avg)
    print("average marks are %d:" %avg)
    for k in range(len(marks)):
           if(marks[k]<40):
               print("failed")
           else:
                print("passed")
print("highest avg marks are %d \n Nmame: %s \n"%(max(avg1),name))

a=[1,2,3,4]
b=[5,6,7,8]
c=[]
for i in range(len(a)):
    for j in range(len(b)):
     if i==j:
         c.append(a[i])
print(c)
list_common=[num for num in a and num in b]
'''