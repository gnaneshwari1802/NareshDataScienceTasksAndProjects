# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:59:09 2019

@author: M GNANESHWARI
"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.learn_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
data_income=pd.read_csv('D:\\DataScience\\income.csv')
data=data_income.copy()
print(data.info())
data.isnull()
#print('Data columns with null values:\n',data.isnull().)
summary_num=data.describe()
print(summary_num)
summary_cate=data.describe(include="0")
print(summary_cate)
data['JobType'].value_counts()
data['occupation'].value_counts()
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
data=pd.read_csv('D:\\DataScience\\income.csv',na_values=["?"])
data.isnull().sum()
missing=data[data.isnull().any(axis=1)]
data2=data.dropna(axis=0)
correlation=data2.corr()
data2.columns
gender=pd.crosstab(index=data2["gender"],columns='count',normalize=True)
print(gender)
gender_salstat=pd.crosstab(index=data2["gender"],margins=True,normalize='index',columns=data2['SalStat'])
print(gender_salstat)
SalStat=sns.countplot(data2['SalStat'])
sns.displot(data2['age'],bins=10,kde=False)
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()
columns_list=list(new_data.columns)
print(columns_list)
features=list(set(columns_list).set(['SalStat']))
print(features)
y=new_data['SalStat'].values
print(y)
x=new_data[features].vallues
print(x)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_start)
logistic=logisticRegression()
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_
prediction=logistic.predict(test_x)
print(prediction)
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
print('misclessified samples: %d' %(test_y is prediciton).sum())
#data2['SalStat']=data2['SalStat'].map({'',''})
print(data2['SalStat'])
cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)
new_data=pd.get_dummies(new_data,drop_first=True)
columns_list=list(new)