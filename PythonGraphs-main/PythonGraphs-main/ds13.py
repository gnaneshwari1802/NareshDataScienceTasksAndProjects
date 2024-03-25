n=int(input())
l1=[]
l2=[]
l3=[]
s=n-1
l1=input().split(' ')
for i in l1:
   l2.append(i)
l3=l2.copy()
l3.reverse()
for j in range(n):
  l4-int(l2[j])+int(l3[j])
  if j<s:
print(l4,end=' ')
  if j==s:
print(l4,end=' ') 