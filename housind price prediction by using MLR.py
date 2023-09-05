# Problem Statement:

## Consider a real estate company that has a dataset containing the prices of properties in a perticular region. It wishes to use the data to predict the sale prices of the properties based on important factors such as area, bedrooms, parking, etc.

# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# Import the data set

df = pd.read_csv(r"D:\Data science studies\datasets\Housing_pricing.csv")

## what is shape of dataset ?

print('This data set having ',df.shape[0],' number of rows and ',df.shape[1],' number of columns')

## 1st 5 rows of dataset

df.head()

# Checking null values

df.isnull().sum()

# unique values of each dataset

df.nunique()

## infermation about dataset

df.shape

df.info()

df.describe()

# Data cleaning

df.isnull().sum()*100/df.shape[0]

### Dataset donot have any missing values so this is cleaned data.

## data types of dataset

df.dtypes

# Outlier Analysis


fig, axs = plt.subplots(2,3, figsize = (10,5))
sns.boxplot(df['price'], ax = axs[0,0])
sns.boxplot(df['area'], ax = axs[0,1])
sns.boxplot(df['bedrooms'], ax = axs[0,2])
sns.boxplot(df['bathrooms'], ax = axs[1,0])
sns.boxplot(df['stories'], ax = axs[1,1])
sns.boxplot(df['parking'], ax = axs[1,2])
fig.suptitle("outlier analysis")
plt.tight_layout()

### Here we can see price and area has small amount of outliers so we will try to remove those outliers

# Remove outliers

Q1,Q3=df['price'].quantile([0.25,0.75])

IQR=Q3-Q1
LL=Q1-1.5*IQR
UL=Q3+1.5*IQR

IQR

df = df[(df['price']>=LL) & (df['price']<=UL)]

plt.figure(figsize=(5,3))
sns.boxplot(df['price'])
plt.show()

Q1,Q3=df['area'].quantile([0.25,0.75])

IQR=Q3-Q1
LL=Q1-1.5*IQR
UL=Q3+1.5*IQR

IQR

df = df[(df['area']>=LL) & (df['area']<=UL)]

plt.figure(figsize=(5,3))
sns.boxplot(df['area'])
plt.show()

fig, axs = plt.subplots(2,3, figsize = (10,5))
sns.boxplot(df['price'], ax = axs[0,0])
sns.boxplot(df['area'], ax = axs[0,1])
sns.boxplot(df['bedrooms'], ax = axs[0,2])
sns.boxplot(df['bathrooms'], ax = axs[1,0])
sns.boxplot(df['stories'], ax = axs[1,1])
sns.boxplot(df['parking'], ax = axs[1,2])
fig.suptitle("outlier analysis")
plt.tight_layout()

# EDA

sns.pairplot(df)
plt.show()

sns.pairplot(df,hue = 'furnishingstatus' )
plt.show()

# How price varies with categorical attributes

plt.figure(figsize=(10,5))

plt.subplot(3,3,1)
sns.boxplot(x = 'price', y = 'mainroad' , data = df)

plt.subplot(3,3,2)
sns.boxplot(x = df['price'], y = df['guestroom'] , data = df)

plt.subplot(3,3,3)
sns.boxplot(x = df['price'], y = df['basement'] , data = df)

plt.subplot(3,3,4)
sns.boxplot(x = df['price'], y = df['hotwaterheating'] , data = df)

plt.subplot(3,3,5)
sns.boxplot(x = df['price'], y = df['airconditioning'] , data = df)

plt.subplot(3,3,6)
sns.boxplot(x = df['price'], y = df['prefarea'] , data = df)

plt.subplot(3,3,7)
sns.boxplot(x = df['price'], y = df['furnishingstatus'] , data = df)

plt.tight_layout()

# How price varies with numerical attributes

plt.figure(figsize=(3,2))
sns.lmplot(x = 'price', y = 'area' , data = df)
plt.show()

sns.lmplot(x = 'price', y = 'bedrooms' , data = df)
plt.show()

sns.lmplot(x = 'price', y = 'bathrooms', data = df)
plt.show()

sns.lmplot(x = 'price', y = 'stories' , data = df)
plt.show()

sns.lmplot(x = 'price', y = 'parking' , data = df)
plt.show()



# need to convert all categorical data into yes = 1 and No = 0 form

df.head()

df.dtypes

cat_col =['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']

maping={'no':0,'yes':1}

def maping (x):
    return x.map({"no":0,"yes":1})

df[cat_col] = df[cat_col].apply(maping)

df[cat_col].head()

df.head()

df['furnishingstatus']=df['furnishingstatus'].map({"unfurnished":0,"furnished":1,"semi-furnished":2})

dummy = pd.get_dummies(df['furnishingstatus'])

dummy

df = pd.concat([df,dummy],axis=1)

df.columns

df.drop(['furnishingstatus'],axis = 1,inplace=True)

# shape of df after removing outlier

print('After removing outlier This data set having ',df.shape[0],' number of rows and ',df.shape[1],' number of columns')

df.head()


# checking co relation between attribute

plt.figure(figsize = (16, 10))
sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")
plt.show()

## herw we can see price and area 53% corelated

# Split data into X(Attributes) and y(target)

X = df.drop(['price'],axis=1)
X.head()

df.shape

X.shape

y= df['price']

y

print(y.shape)

# Train test Split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

X_train

X_test

y_train

y_test

print('shape of X_train is',X_train.shape)
print('shape of y_train is',y_train.shape)

print('shape of X_test is',X_test.shape)
print('shape of y_test is',y_test.shape)

# model build

# Simple linear regression

from sklearn.linear_model import LinearRegression

lg = LinearRegression()

lg.fit(X_train,y_train)

y_pred = lg.predict(X_test)

y_pred

len(y_pred)

# slope is generrated from linear regress algorith which fit to dataset 

m = lg.coef_

m

# interceppt also generatre by model. 

c = lg.intercept_

c

# checking r2 score of model

from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)

## hence the performance of our model is 65.36%

# checking bias and variance

bias = lg.score(X_train, y_train)

variance = lg.score(X_test, y_test)

print(bias)
print(variance)

import statsmodels.api as sm

X = np.append(arr = np.ones((517,1)).astype(int), values = X, axis = 1) 
X

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]  # we have removed last column here.
X_opt

lg_OLS = sm.OLS(endog=y, exog=X_opt).fit()

lg_OLS.summary()



X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]  # we have removed last column here.
X_opt

lg_OLS = sm.OLS(endog=y, exog=X_opt).fit()

lg_OLS.summary()




X_opt = X[:,[0,1,3,4,5,6,7,8,9,10,11,12]]  # we have removed last column here.
X_opt

lg_OLS = sm.OLS(endog=y, exog=X_opt).fit()

lg_OLS.summary()



X_opt = X[:,[0,1,3,4,5,7,8,9,10,11,12]]  # we have removed last column here.
X_opt

lg_OLS = sm.OLS(endog=y, exog=X_opt).fit()

lg_OLS.summary()

