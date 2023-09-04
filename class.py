# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:05:30 2023

@author: M GNANESHWARI
"""


import numpy as np 	#Array		
import matplotlib.pyplot as plt		
import pandas as pd		
#--------------------------------------------

# import the dataset & divided my dataset into independe & dependent

dataset = pd.read_csv(r"C:\Users\M GNANESHWARI\Desktop\Data.csv")

x = dataset.iloc[:, :-1].values	

y = dataset.iloc[:,3].values  
x
y
#--------------------------------------------
#In statistics, imputation is the process of replacing missing data with substituted values. When substituting for a data point, it is known as "unit imputation"; when substituting for a component of a data point, it is known as "item imputation".
from sklearn.impute import SimpleImputer # SPYDER 4 

imputer = SimpleImputer() 
imputer
#What does Imputer fit () do?
#If you tell the Imputer that you want the mean of all the values in the column to be used to replace all the NaNs in that column, the Imputer has to calculate the mean first. This step of calculating that value is called the fit() method
imputer = imputer.fit(x[:,1:3]) 
imputer
#The fit() method helps in fitting the training dataset into an estimator (ML algorithms). The transform() helps in transforming the data into a more suitable form for the model.
x[:, 1:3] = imputer.transform(x[:,1:3])
x
# Fit the imputer on the data and transform the data to impute missing values
#imputed_data = imputer.fit_transform(data)
from sklearn.impute import SimpleImputer # spyder

#imputer = SimpleImputer(missing_values=np.nan, strategy ="median")

# --------------------------------------------------------------------
#imputer = SimpleImputer(missing_values=np.nan, strategy ="mean")
imputer = SimpleImputer(missing_values=np.nan, strategy ="most_frequent")
imputer = imputer.fit(x[:,1:3])

x[:, 1:3] = imputer.transform(x[:,1:3])
x
#--------------------------------------------------------------------

# How to Encode Categorical data & crete a dummy variable
#The code snippet you've provided imports the LabelEncoder class from scikit-learn's sklearn.preprocessing module. LabelEncoder is a commonly used tool for encoding categorical labels (textual or nominal data) into numerical values, which are often required for machine learning algorithms that expect numerical input.
from sklearn.preprocessing import LabelEncoder
#Encode target labels with value between 0 and n_classes-1. This transformer should be used to encode target values, i.e. y , and not the input X .
"""What does preprocessing LabelEncoder do?
LabelEncoder can be used to normalize labels. It can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels. Fit label encoder.
"""
labelencoder_X = LabelEncoder()
"""
What does sklearn Fit_transform do?
The fit_transform method from the sklearn. preprocessing class is used to preprocess the data for model training. The fit() method calculates the various required parameters, and the transform() method applies the calculated parameters to standardize the data.
"""
x[:,0]=labelencoder_X.fit_transform(x[:,0])
labelencoder_X
x[:,0]
#------------------------------------------------------------

labelencoder_y =LabelEncoder()

y = labelencoder_y.fit_transform(y)
y


#------------------------------------------------------------

#SPLITING THE DATASET IN TRAINING SET & TESTING SET

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y, test_size = 0.3)
X_train
X_test
y_train
y_test
