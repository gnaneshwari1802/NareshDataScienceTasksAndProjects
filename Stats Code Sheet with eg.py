# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:07:45 2023

@author: M GNANESHWARI
"""

import pandas as pd
import numpy as np
df = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\11th\11th\SIMPLE LINEAR REGRESSION\Salary_Data.csv')
df.mean() # this will give mean of entire dataframe 
df['Salary'].mean() # this will give us mean of that particular column 
df.median() # this will give median of entire dataframe 
df['Salary'].median() # this will give us median of that particular column 
df['Salary'].mode() # this will give us mode of that particular column 
df.var() # this will give variance of entire dataframe 
df['Salary'].var() # this will give us variance of that particular column
df.std() # this will give standard deviation of entire dataframe 
df['Salary'].std() # this will give us standard deviation of that particular column
# for calculating cv we have to import a library first
from scipy.stats import variation
variation(df.values) # this will give cv of entire dataframe 
variation(df['Salary']) # this will give us cv of that particular column
df.corr() # this will give correlation of entire dataframe
df['Salary'].corr(df['YearsExperience']) # this will give us correlation between these t
df.skew() # this will give skewness of entire dataframe
df['Salary'].skew() # this will give us skewness of that particular column
df.sem() # this will give standard error of entire dataframe 
df['Salary'].sem() # this will give us standard error of that particular column
# for calculating Z-score we have to import a library first
import scipy.stats as stats
df.apply(stats.zscore) # this will give Z-score of entire dataframe
stats.zscore(df['Salary']) # this will give us Z-score of that particular column
a = df.shape[0] # this will gives us no.of rows
b = df.shape[1] # this will give us no.of columns
degree_of_freedom = a-b
print(degree_of_freedom) # this will give us degree of freedom for entire dataset
#First we have to separate dependent and independent variables
X=df.iloc[:,:-1].values #independent variable
y=df.iloc[:,1].values # dependent variable
y_mean = np.mean(y) # this will calculate mean of dependent variable
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_predict = reg.predict(X_test) # before doing this we have to train,test and split our
SSR = np.sum((y_predict-y_mean)**2)
print(SSR)
#First we have to separate dependent and independent variables
X=df.iloc[:,:-1].values #independent variable
y=df.iloc[:,1].values # dependent variable
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_predict = reg.predict(X_test) # before doing this we have to train,test and split our
y = y[0:6]
SSE = np.sum((y-y_predict)**2)
print(SSE)
mean_total = np.mean(df.values) # here df.to_numpy()will convert pandas Dataframe to Nump
SST = np.sum((df.values-mean_total)**2)
print(SST)
r_square = SSR/SST
r_square