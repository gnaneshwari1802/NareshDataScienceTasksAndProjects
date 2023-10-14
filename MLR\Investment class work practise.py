# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 12:11:13 2023

@author: M GNANESHWARI
"""

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv(r"C:\Users\M GNANESHWARI\Desktop\14th\14th\MLR\Investment.csv")
dataset
X = dataset.iloc[:, :-1]

y = dataset.iloc[:, 4]

X= pd.get_dummies(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

slope = regressor.coef_
slope

cons = regressor.intercept_
cons

bias = regressor.score(X_train, y_train)
bias #95%

variance = regressor.score(X_test, y_test)
variance #95%


# **** we build the model so far

import statsmodels.formula.api as sm



# Create a simple linear regression model using the ols function
dataset.rename(columns={'HR Domain': 'HR_Domain', 'Marketing Spend': 'Marketing_Spend'}, inplace=True)

model = sm.ols(formula='Profit~HR_Domain+Marketing_Spend+State',data=dataset).fit()

# Print the summary of the regression results
print(model.summary())
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)



X_opt = X[:,[0,1,2,3,4,5]]

#OrdinaryLeastSquares
regressor_OLS = sm.ols(formula='Profit~HR_Domain+Marketing_Spend+State',data=dataset).fit()

regressor_OLS.summary()


X_opt = X[:,[0,1,2,3,5]]

#OrdinaryLeastSquares
regressor_OLS = sm.ols(formula='Profit~HR_Domain+Marketing_Spend+State',data=dataset).fit()

regressor_OLS.summary()



X_opt = X[:,[0,1,2,3]]

#OrdinaryLeastSquares
regressor_OLS = sm.ols(formula='Profit~HR_Domain+Marketing_Spend+State',data=dataset).fit()

regressor_OLS.summary()

X_opt = X[:, [0,1,3]]
regressor_OLS = sm.ols(formula='Profit~HR_Domain+Marketing_Spend+State',data=dataset).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1]]
regressor_OLS = sm.ols(formula='Profit~HR_Domain+Marketing_Spend+State',data=dataset).fit()
regressor_OLS.summary()

'''
# data sceince you will inform to the ceo 
pleas spend on research part and lets wait for the result 
- works thats good ( 3mont)
-- change the model with more (50) 500
'''
