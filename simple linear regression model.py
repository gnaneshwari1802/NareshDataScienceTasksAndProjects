# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:39:45 2023

@author: M GNANESHWARI
"""

# import require library 

import numpy as np 	

import matplotlib.pyplot as plt

import pandas as pd	

# import the dataset


dataset = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\11th\11th\SIMPLE LINEAR REGRESSION\Salary_Data.csv')

# split the data to independent variable 
X = dataset.iloc[:, :-1].values

# split the data to dependent variabel 
y = dataset.iloc[:,1].values 

# as d.v is continus that regression algorithm 
# as in the data set we have 2 attribute we slr algo

# split the dataset to 80-20%
from sklearn.model_selection import train_test_split
model_selection is used to split our data into train and test sets where feature variables are given as input in the method. test_size determines the portion of the data which will go into test sets and a random state is used for data reproducibility
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
What is Model_selection train_test_split?
The train_test_split function of the sklearn. model_selection package in Python splits arrays or matrices into random subsets for train and test data, respectively.
As a high-level library, it lets you define a predictive data model in just a few lines of code, and then use that model to fit your data. It's versatile and integrates well with other Python libraries, such as matplotlib for plotting, numpy for array vectorization, and pandas for dataframes.
#we called simple linear regression algoriytm from sklearm framework 
from sklearn.linear_model import LinearRegression
The scikit-learn library in Python implements Linear Regression through the LinearRegression class. This class allows us to fit a linear model to a dataset, predict new values, and evaluate the model's performance. To use the LinearRegression class, we first need to import it from sklearn. linear_model module.
regressor = LinearRegression()

# we build simple linear regression model regressor
regressor.fit(X_train, y_train)


# test the model & create a predicted table 
y_pred = regressor.predict(X_test)

# visualize train data point ( 24 data)
plt.scatter(X_train, y_train, color = 'red') 
matplotlib.pyplot.scatter()
Scatter plots are used to observe relationship between variables and uses dots to represent the relationship between them. The scatter() method in the matplotlib library is used to draw a scatter plot. Scatter plots are widely used to represent relation among variables and how change in one affects the other. 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
  The plot() function is used to draw points (markers) in a diagram. By default, the plot() function draws a line from point to point. The function takes parameters for specifying points in the diagram. Parameter 1 is an array containing the points on the x-axis.
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visulaize test data point 
plt.scatter(X_test, y_test, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# slope is generrated from linear regress algorith which fit to dataset 
m = regressor.coef_

# interceppt also generatre by model. 
c = regressor.intercept_

# predict or forcast the future the data which we not trained before 
y_12 = 9312 * 12 + 26780
y_12

y_20 = 9312 * 20 + 26780
y_20


# to check overfitting  ( low bias high variance)
bias = regressor.score(X_train, y_train)
bias


# to check underfitting (high bias low variance)
variance = regressor.score(X_test,y_test)
variance


# deployment in flask & html 
# mlops (azur, googlcolab, heroku, kubarnate)
Simple Linear Regression with Experience -Salary Dataset
split ratio(80%-20%, 70%-30%, 75%-25%)
