# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 18:31:19 2023

@author: M GNANESHWARI
"""
#House Prices using Backward Elimination
#Just started with machine learning. I have used backward Elimination to check the usefulness of dependent variables.
#importing all the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%matplotlib inline

#importing dataset using panda
dataset = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\MLR-main\MLR-main\house_data.csv')
#to see what my dataset is comprised of
dataset.head()
#checking if any value is missing
print(dataset.isnull().any())
#checking for categorical data
print(dataset.dtypes)
#dropping the id and date column
dataset = dataset.drop(['id','date'], axis = 1)
#understanding the distribution with seaborn
"""
The Seaborn.plotting_context() method gets the parameters that control the scaling of plot elements. The scaling does not effect the overall style of the plot but it does affect things like labels, lines and other elements of the plot. It is done using the matplotlib rcParams system.

For example, if the base context is â€œnotebookâ€ and other contexts are â€œposterâ€, â€œpaperâ€ and â€œinformationâ€. These other contexts are the notebook parameters scaled by different values. Font elements in a context can also be scaled independently.

This function can also be used to alter the global default values.

Syntax
Following is the syntax of the plotting_context() method −

seaborn.plotting_context(context=None, font_scale=1, rc=None)
Parameters
Following are the parameters of this method −

S.No	Parameter and Description
1	context
Takes the following as input none, dict, or one of {paper, notebook, talk, poster} and determines a dictionary of parameters or the name of a preconfigured set.

2	Rc
Takes rcdict as value and is an optional parameter that performs Parameter mappings to override the values in the preset seaborn style dictionaries. This only updates parameters that are considered part of the style definition.

3	Font_scale
Takes a floating point value as input, and is optional parameter. It separate scaling factor to independently scale the size of the font elements.

"""
with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']],hue='bedrooms', palette='tab20',size=6)
g.set(xticklabels=[]);
#separating independent and dependent variable
X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values
#splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) 

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#Backward Elimination
import statsmodels.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((21613,19)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
"""
The code you provided appears to implement a backward elimination method for feature selection in linear regression. The purpose of this code is to iteratively remove features (independent variables) from the input data x based on their p-values until the significance level (SL) is reached or until no more features can be removed without reducing the adjusted R-squared value.

Here is an overview of how the code works:

It initializes some variables, including numVars (the number of independent variables), and creates a temporary array temp to store removed features.

It enters a loop that iterates over the independent variables one by one.

Inside the loop:

It fits an OLS (Ordinary Least Squares) regression model (regressor_OLS) using the current set of independent variables x.
It calculates the maximum p-value among the coefficients in the model.
It calculates the adjusted R-squared value before removing any features.
If the maximum p-value is greater than the significance level (SL), it enters another loop to identify the variable with the maximum p-value and removes it from x. This is done using the np.delete function.
It fits a new OLS regression model (tmp_regressor) with the updated x and calculates the adjusted R-squared value after the removal.
It compares the adjusted R-squared value before and after removal. If the adjusted R-squared value has not decreased, it means that removing the feature did not significantly affect the model's performance, so the code continues removing more features.
If removing the feature decreased the adjusted R-squared value, the code rolls back the last removal by adding the removed feature back to x and returns the updated x with the feature rollback.
After the loop completes, the code prints the summary of the final regressor_OLS model and returns the updated x with the selected features.

It's worth noting that the code does not explicitly define y, which should be the dependent variable for the linear regression model. Additionally, it does not include import statements for the required libraries (np and sm) or the necessary library initializations. Make sure to import the required libraries and provide the y variable before using this code.
"""

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]
X_Modeled = backwardElimination(X_opt, SL)
