# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:48:41 2023

@author: M GNANESHWARI
"""
"""
Featurization, Model Selection & Tuning - Linear Regression
Why is regularization required ?

We are well aware of the issue of 'Curse of dimensionality', where the no. of columns are so huge that the no. of rows does not cover all the permutation and combinations that is applicable for this dataset. For eg: Data having 10 columns should have 10! rows but it has only 1000 rows

Therefore,when we depict this graphically there would be lot of white spaces as the datapoints for those regions may not be covered in the dataset.

If a linear regression model is tested over such a data, the model will tend to overfit this data by having sharp peaks & slopes. Such a model would have 100% training accuracy but would definitely fail in the test environment.

Thus arose the need of introducing slight errors in the form of giving smooth bends instead of sharp peaks (thereby reducing overfit).This is achieved by tweaking the model parameters (coefficients) and the hyperparameters (penalty factor).

Agenda
Perform basic EDA
Scale data and apply Linear, Ridge & Lasso Regression with Regularization
Compare the r^2 score to determine which of the above regression methods gives the highest score
Compute Root mean squared error (RMSE) which inturn gives a better score than r^2
Finally use a scatter plot to graphically depict the correlation between actual and predicted mpg values
1. Import packages and observe dataset

"""
#Import numerical libraries
import pandas as pd
import numpy as np

#Import graphical plotting libraries
import seaborn as sns
Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. For a brief introduction to the ideas behind the library, you can read the introductory notes or the paper.
import matplotlib.pyplot as plt
What is matplotlib Pyplot in Python?
Pyplot is an API (Application Programming Interface) for Python's matplotlib that effectively makes matplotlib a viable open source alternative to MATLAB. Matplotlib is a library for data visualization, typically in the form of plots, graphs and charts.
#%matplotlib inline

#Import Linear Regression Machine Learning Libraries
from sklearn import preprocessing
What is Sklearn in Python?
Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.
from sklearn.preprocessing import PolynomialFeatures
What is polynomial features in sklearn?
In simple terms, PolynomialFeatures is a method for feature engineering that creates new features by raising the existing features to a power. For example, if we have a feature x, we can create a new feature x^2 by squaring x. We can also create a new feature x^3 by cubing x, and so on.
The sklearn. preprocessing package provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators. In general, learning algorithms benefit from standardization of the data set.
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
What is r2_score in Python?
R^2 (coefficient of determination) regression score function. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.
Why use sklearn metrics?
Scikit learn metrics are used to implement the function assessing the prediction error for a specified purpose. Multiple types of scikit learn metrics are available in the module of sklearn. The module of sklearn metrics implements several utility functions and measures the classification's performance.
What is ridge in Python?
Understanding Ridge Regression Using Python - Shiksha Online
Ridge regression is a regularization technique that penalizes the size of the regression coefficient based on the l2 norm. It is also known as L2 regularization. It is used to eliminate multicollinearity in models. Suitable for the dataset that has a higher number of predictor variables than the number of observations.
Lasso regression penalizes less important features of your dataset and makes their respective coefficients zero, thereby eliminating them. Thus it provides you with the benefit of feature selection and simple model creation. So, if the dataset has high dimensionality and high correlation, lasso regression can be used.
Lasso regression includes a regularization penalty in the objective function, which helps prevent overfitting and improve the model’s generalization performance. The penalty shrinks the coefficients towards zero, resulting in a simpler model that is less prone to overfitting.
data = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\17th\17th\lasso, ridge, elastic net\TASK-22_LASSO,RIDGE\car-mpg.csv')
data.head()

#Drop car name
#Replace origin into 1,2,3.. dont forget get_dummies
#Replace ? with nan
#Replace all nan with median

data = data.drop(['car_name'], axis = 1)
data['origin'] = data['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
data = pd.get_dummies(data,columns = ['origin'])
data = data.replace('?', np.nan)
data = data.apply(lambda x: x.fillna(x.median()), axis = 0)
Pandas.apply allow the users to pass a function and apply it on every single value of the Pandas series. It comes as a huge improvement for the pandas library as this function helps to segregate data according to the conditions required due to which it is efficiently used in data science and machine learning.
apply() method. This function acts as a map() function in Python. It takes a function as an input and applies this function to an entire DataFrame. If you are working with tabular data, you must specify an axis you want your function to act on ( 0 for columns; and 1 for rows).
    data.head() 
"""
We have to predict the mpg column given the features.

2. Model building
Here we would like to scale the data as the columns are varied which would result in 1 column dominating the others.

First we divide the data into independent (X) and dependent data (y) then we scale it.

Tip!:
*The reason we don't scale the entire data before and then divide it into train(X) & test(y) is because once you scale the data, the type(data_s) would be numpy.ndarray. It's impossible to divide this data when it's an array.

Hence we divide type(data) pandas.DataFrame, then proceed to scaling it.
"""
X = data.drop(['mpg'], axis = 1) # independent variable
y = data[['mpg']] #dependent variable
#Scaling the data

X_s = preprocessing.scale(X)
What is scale () in Python?
The scale() function is an inbuilt function in the Python Wand ImageMagick library which is used to change the image size by scaling each pixel value by given columns and rows. Syntax: scale(columns, rows)
X_s = pd.DataFrame(X_s, columns = X.columns) #converting scaled data into dataframe

y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y_s, columns = y.columns) #ideally train, test data should be in columns
#Split into train, test set

X_train, X_test, y_train,y_test = train_test_split(X_s, y_s, test_size = 0.30, random_state = 1)
X_train.shape
#2.a Simple Linear Model
#Fit simple linear model and find coefficients
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
Often, when dealing with iterators, we also need to keep a count of iterations. Python eases the programmers’ task by providing a built-in function enumerate() for this task. The enumerate () method adds a counter to an iterable and returns it in the form of an enumerating object. This enumerated object can then be used directly for loops or converted into a list of tuples using the list() function.
for idx, col_name in enumerate(X_train.columns):
    print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx]))
    
intercept = regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))
"""
2.b Regularized Ridge Regression
#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff
"""
ridge_model = Ridge(alpha = 0.3)
ridge_model.fit(X_train, y_train)

print('Ridge model coef: {}'.format(ridge_model.coef_))
#As the data has 10 columns hence 10 coefficients appear here    
"""
2.c Regularized Lasso Regression
#alpha factor here is lambda (penalty term) which helps to reduce the magnitude of coeff
"""
lasso_model = Lasso(alpha = 0.1)
lasso_model.fit(X_train, y_train)

print('Lasso model coef: {}'.format(lasso_model.coef_))
#As the data has 10 columns hence 10 coefficients appear here   

"""
Here we notice many coefficients are turned to 0 indicating drop of those dimensions from the model

3. Score Comparison
#Model score - r^2 or coeff of determinant
#r^2 = 1-(RSS/TSS) = Regression error/TSS 


#Simple Linear Model
"""
print(regression_model.score(X_test, y_test))

print('*************************')
#Ridge
print(ridge_model.score(X_train, y_train))
print(ridge_model.score(X_test, y_test))

print('*************************')
#Lasso
print(lasso_model.score(X_train, y_train))
print(lasso_model.score(X_test, y_test))
"""
Polynomial Features
If you wish to further compute polynomial features, you can use the below code.

#poly = PolynomialFeatures(degree = 2, interaction_only = True)

#Fit calculates u and std dev while transform applies the transformation to a particular set of examples
#Here fit_transform helps to fit and transform the X_s
#Hence type(X_poly) is numpy.array while type(X_s) is pandas.DataFrame 
#X_poly = poly.fit_transform(X_s)
#Similarly capture the coefficients and intercepts of this polynomial feature model
"""
"""
4. Model Parameter Tuning
r^2 is not a reliable metric as it always increases with addition of more attributes even if the attributes have no influence on the predicted variable. Instead we use adjusted r^2 which removes the statistical chance that improves r^2
(adjusted r^2 = r^2 - fluke)

Scikit does not provide a facility for adjusted r^2... so we use statsmodel, a library that gives results similar to what you obtain in R language
This library expects the X and Y to be given in one single dataframe
"""
data_train_test = pd.concat([X_train, y_train], axis =1)
data_train_test.head()
import statsmodels.formula.api as smf
ols1 = smf.ols(formula = 'mpg ~ cyl+disp+hp+wt+acc+yr+car_type+origin_america+origin_europe+origin_asia', data = data_train_test).fit()
ols1.params
print(ols1.summary())
#Lets check Sum of Squared Errors (SSE) by predicting value of y for test cases and subtracting from the actual y for the test cases
mse  = np.mean((regression_model.predict(X_test)-y_test)**2)

# root of mean_sq_error is standard deviation i.e. avg variance between predicted and actual
import math
rmse = math.sqrt(mse)
print('Root Mean Squared Error: {}'.format(rmse))
"""
So there is an avg. mpg difference of 0.37 from real mpg

# Is OLS a good model ? Lets check the residuals for some of these predictor.
"""
fig = plt.figure(figsize=(10,8))
sns.residplot(x= X_test['hp'], y= y_test['mpg'], color='green', lowess=True )


fig = plt.figure(figsize=(10,8))
sns.residplot(x= X_test['acc'], y= y_test['mpg'], color='green', lowess=True )
# predict mileage (mpg) for a set of attributes not in the training or test set
y_pred = regression_model.predict(X_test)

# Since this is regression, plot the predicted y value vs actual y values for the test data
# A good model's prediction will be close to actual leading to high R and R2 values
#plt.rcParams['figure.dpi'] = 500
plt.scatter(y_test['mpg'], y_pred)
"""
5. Inference
Both Ridge & Lasso regularization performs very well on this data, though Ridge gives a better score. The above scatter plot depicts the correlation between the actual and predicted mpg values.

*This kernel is a work in progress.*
"""
