# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:46:05 2023

@author: M GNANESHWARI
"""
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Updated import
from sklearn.linear_model import LinearRegression
# Set the threshold to infinity to always print the full array
# Any results you write to the current directory are saved as output.
np.set_printoptions(threshold=np.inf)



#directory = r'C:\Users\M GNANESHWARI\Desktop\11th\11th\SLR - Practicle'  # Update this path to the correct directory
directory = 'C:\\Users\\M GNANESHWARI\\Desktop\\11th\\11th\\SLR - Practicle'  # Update this path to the correct directory

# List files in the directory
file_list = os.listdir(directory)

# Print the list of files
for file in file_list:
    print(file)
# Importing DataSet
dataset = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\11th\11th\SLR - Practicle\House_data.csv')
 space = dataset['sqft_living']
price = dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

# Splitting the data into Train and Test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=1/3, random_state=0)

# Fitting simple linear regression to the Training Set
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# Predicting the prices
pred = regressor.predict(xtest)

# Visualizing the training Test Results
plt.scatter(xtrain, ytrain, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

# Visualizing the Test Results
plt.scatter(xtest, ytest, color='red')
plt.plot(xtest, regressor.predict(xtest), color='blue')  # Updated xtest here
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()
