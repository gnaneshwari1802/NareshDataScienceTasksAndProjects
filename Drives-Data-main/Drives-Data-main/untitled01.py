# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 13:00:10 2023

@author: M GNANESHWARI
"""

# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\EMP SAL.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
X
y

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='poly', degree=7, gamma='auto')
regressor.fit(X, y)

from sklearn.neighbors import KNeighborsRegressor
regressor_knn = KNeighborsRegressor(n_neighbors=6)
regressor_knn.fit(X,y)

#regressor created properly with default parameter after execute the above line of code
#now we will check what was the actual salary after scaling 

# Predicting a new result

y_pred_knn = regressor_knn.predict([[6.5]])


y_pred_svr = regressor.predict([[6.5]])

#y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#we will see what is predicted salary for the 6.5yrs of exp new employee
#always check the next argument function by select the object inspector
#you have to transform the 6.5 numerical value transform and fit to the regressor 
#we have to do the inverse transform to get the orginial scale & by using the inverse_transform then we will get the scaled prediction salary
#after execute we get very great prediction we found the predicted sal is 170k which is too good
#we can say that our svr model quite good model compare to polynomial regression, finally we can say that svr is great model


# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#if you check the output that is svr model & its predicting the each of the real observation
#red points are real observation point & blue lines are predicted line & now you can say svr is fitted much better curve on the dataset
#same hear if you check the ceo actual observation point but you will find as still we can improve the graph and lets see how can we do that in svr
#in this case ceo is outlier hear becuase ceo is quite far from our observation, thats ok

#what exactly we are doing hear to check the what exactly employees have 6.5yrs experience predict salary


# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#great curve you got isn't it , same dataset you worked polynomial regressor & svm regressor 