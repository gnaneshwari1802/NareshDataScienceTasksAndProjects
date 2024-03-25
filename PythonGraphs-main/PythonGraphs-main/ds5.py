# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 20:24:14 2019

@author: M GNANESHWARI


T = int(input())

for i in range(T):
    
    line = int(input())
    ones = zeros = 0
    for c in line:
        
        if int(c) == 1:
            ones += 1
        elif int(c) == 0:
            zeros += 1
        else:
            raise ValueError
        if ones > 1 and zeros > 1:
            print("No")
            break
    if ones == 1:
        print("Yes")
    elif zeros == 1:
        print("Yes")
    elif ones == 0 or zeros == 0:
        print("No")


list1 = [10, 20, 4, 45, 99] 
  
# sorting the list 
list1.sort() 
  
# printing the last element 
print("Largest element is:", list1[-1])        
print("Largest element is:", list1[-2]) 

import pandas as pd

data_farms=pd.read_csv("D:\\DataScience\\farms.csv")
bimas_tab=pd.crosstab(index=data_farms["bimas"],columns=["count"])
status_varie_bim_tab=pd.crosstab(index=data_farms["status"],columns=data_farms["varieties"],index=data_farms["bimas"],margins=True)

print(bimas_tab)
print(status_varie_bim_tab)
#print()*8

import pandas as pd
data_cars=pd.read_csv("D:\\DataScience\\ToyotaCorolla.csv")
Auto_Fuel=pd.crosstab(index=data_cars["Automatic"],columns=data_cars["Fuel_Type"],normalize=True,dropna=True)
print(Auto_Fuel)

#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
year=[1960,1970,1980,1990]
pop_Indonesia=[44.91,58.09,78.07,107.7,138.5,170.6]
pop_india=[449.48,553.57]
plt.plot(year,pop_Indonesia,'b',label='line one',color="o")
plt.plot(year,pop_india,'r',label='line two',color="b")
plt.xlabel("countries")
plt.ylabel("population")
plt.grid(True,color='red')
plt.title("Indonesia and india population till 2010")
plt.show()

from matplotlib import pyplot as plt
x=[5,8,10]
y=[12,16,6]
x2=[6,9,11]
y2=[6,17,5]
plt.plot(x,y,'b',label='line one',linewidth=5)
plt.plot(x2,y2,'r',label='line two',linewidth=5)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('epic info')
plt.legend()
plt.grid(True,color='brown')
plt.show()

from matplotlib import pyplot as plt

Days=[0.25,1.25,2.25,3.25,4.25]
Distance=[50,69,70,84,90,100]
plt.bar([.75,1.75,2.75,3.75,4.75],[80,20,20,50,60],
label="Audi",color='r',width=.5)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Distance')
plt.title('Info')
plt.show()

from matplotlib import pyplot as plt

plt.bar([0.25,1.25,2.25,3.25,4.25],[50,60,70,80,90],
label="bmw",color='b',width=.5)

plt.bar([.75,1.75,2.75,3.75,4.75],[80,20,20,50,60],
label="Audi",color='r',width=.5)
plt.legend()
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Info')
plt.show()

#1.create a one way table on "First_Type" feature
from matplotlib import pyplot as plt
plt
plt.xlabel('x axix')
plt.ylabel('y axix')
plt.title('info')
plt.show()

import numpy as np
#matrix a
a=np.array([[3,7],[5,2]])
#matrix b
b=np.array([27,16])
#ax=b,x=a^-1.b
x=np.linalg.solve(a,b)
print(x)

import numpy as np
A=np.array([[2,1,1],[1,3,2],[1,0,0]])
B=np.array([4,5,6])
print("solution:\n",np.linalg.solve(A,B))

import numpy as np
A=np.array([[2,1,1],[1,3,2],[1,0,0]])
B=np.array([4,5,6]
print(A)
print(np.transpose(A))
print(np.dot(A,np.transpose(A)))
print(np.dot(np.transpose(A),B))

import numpy as np
ordervolume=[16,10,15,12,11]
arr=np.array(ordervolume)
print(arr)
print("mean=",np.mean(arr))

import pandas as pd
s=pd.Series((10,14,15,25,30,45,55),copy=False,dtype=float)
print(s.size)
print(s.axes)
print(s.dtypes)
print(s.at)
print(s.values)
print(s.shape)
print(s.loc[:-1])
print(s.ftypes)

import pandas as pd
s=pd.Series((10,14,15,25.674,30.45,55,55))
s1=pd.Series((10,14,15,None,30.45,55,55))
s2=pd.Series(('kmit','ngit','kmes'))
print(s.abs())
print(s.add(s1))
print(s2.add_suffix(1))

import matplotlib.pyplot as p
import pandas as pd
girls_grades=[89,90,0,89,100,80,90,150,80,34]
boys_grades=[30,29,49,250,48,38,45,20,30]
grades_range=[10,20,30,40,50,60,70,80,90,100]
p.scatter(grades_range,girls_grades,color='r')
p.scatter(grades_range,boys_grades,color='g')
p.xlabel('grades range')
p.ylabel('grades scored')
p.show()

import pandas as pd
import matplotlib.pyplot as p
import seaborn as sns
data_cars=pd.read_csv("D:\\DataScience\\ToyotaCorolla.csv")
sns.distplot(data_cars["Age_08_04"],kde=False,bins=5)
g=p.gca()
g.set_title("Age distance")

import pandas as pd
#import matplotlib.pyplot as plt
data_loan=pd.read_csv("D:\DataScience\loan.csv")
print(data_loan.isnull().sum())
data_loan["Credit Score"].fillna(data_loan["Credit Score"].mean().inplace=True)
data_loan["Annual Income"].fillna(data_loan["Annual Income"].mean().inplace=True)
data_loan["Backruptcies"].dropna(inplace=True)
print(data_loan.isnull().sum())
print(data_loan["Years in current job"].head())
#data_loan["Years in current job"]=data_loan["Years in current job"].replace('[+(a-z)','',regex=True])
data_loan["Years in current job"]=data_loan["Years in current job"].astype("float")
#print(data_loan[])

import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("D:\DataScience\loan.csv")
dup_Anual_inc=data.duplicated(["Anual Income"])
print(dup_Anual_inc.sum())
data=pd.read_csv("D:\DataScience\loan.csv")
data_skew=data.skew()
data.plot(alpha=0.5,figsize=(2,4))
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv("D:\DataScience\loan.csv")
data_log=np.log(data["Annual Income"])
data_log.plot.hist(alpha=2)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv("D:\DataScience\loan.csv")
data_log=np.sqrt(data["Annual Income"])
data_log.plot.hist(alpha=2)
plt.show()

n,m=input().split(" ")
a=int(n)
b=int(m)
m=[]
for i in range(a):
    c=[]
    for j in range(b):
        c.append(int(input()))
        print(end=" ")
    m.append(a)

from skeleton.linear_model import LogisticRegeression
from skeleton.metrics import accuracy_score
data=pd.read_csv("income.csv",na_values=["?"])
data2=data.dropna(axis=0)
print(data2["SalStat"].value_counts())
column_list=list(new_data.columns)
print(column_list)
features=list(set(column_list)-set(["SalStat"]))
print(features)
#values is used for returning list of values in the list object
y=new_data["SalStat"].values
print(y)          
x=new_data[features].values
print(x)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
#logistic regression uses sigmoid function

import seaborn as s
import pandas as p
from sklearn import LogisticRegression
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
logistic=LogisticRegression()
logistic.fit(train_x,train_y)
print(logistic.coef_)
print(logistic.intercept_)
prediction=logistic.predict(test_x)
print(prediction)
confus_matrix=confusion_matrix(test_y,prediction)
print(confus_matix)
accur_score=accuracy_score(test_y,prediction)
print(accur_score)
print("misclassified")
print(test_y!=prediction).sum())
  
#lambda function maps the 
import seaborn as s
import pandas as p
data_cars=p.read_csv("D:\\DataScience\\cars_sampled.csv")
print(data_cars)
cars_data=data_cars.copy()
print(cars_data.info())
print(cars_data.describe())
p.set_option("display.float_format",lambda x:"%.3f"%x)
print(cars_data.describe())
p.set_option("display.max_columns",500)
print(cars_data.describe())
col=["name","dataCrawled","dataCreated","postalCode","lastSeen"]
cars_data=cars_data.drop(columns=col,axis=1)
print(cars_data)
cars_data.drop_duplicates(Keep="first",inplace="True")
print(cars_data)
print(cars_data.isnull().sum())
yearwise_count=cars_data["yearOfRegistration"].value_count
print(yearwise_count)
print(sum(cars_data["yearOfRegistration"]>2018)
print(sum(cars_data["yearOfRegistration"]<1950)
sns.regplot(x="yearOfRegistration",y="price",scatter=True,fit_reg=False,data=cars_data)
price_count=cars_data["price"].value_counts().sort_index>
print(cars_data["price"]>150000)
print(cars_data["price"]<100)
power_count=cars_data["price"].value.counts().sort_index
print(cars_data["powerPS"].value_counts().sort_)
print(cars_data["powerPS"].describe())
print(sum(cars_data["powrePS"]>500))
print(sum(cars_data["powerPS"]<10))
cars_data=cars_data[(cars_data.yearOfRegistration<=2018)
&(cars_data.yearOfRegistration>=1950)
&(cars_data.price>=100)
&(cars_data.price<=150000)
&(cars_data.powerPS>=10)
&(cars_data.powerPS<=500)]
print(cars_data)
cars_data["monthOfRegistration"]/=12
cars_data["Age"]=(2018.cars_data["yearOfRegistration"])+
cars_data["Age"]=round(cars)_data["Age"],2)
print(cars_data["Age"].describe())
cars_data.drop(columns=["yearOfRegistration","monthOfRegistration"])
s.distplot(cars_data["Age"])
s.boxplot(cars_data["Age"])
s.distplot(cars_data["price"])
s.boxplot(cars_data["price"])
s.distplot(cars_data["powerPS"])
s.boxplot(cars_data["powerPS"])
s.regplot(x="Age",y="price)
"""