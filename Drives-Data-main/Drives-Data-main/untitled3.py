# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 20:50:26 2023

@author: M GNANESHWARI
"""

# Random Forest

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds = pd.read_csv(r'D:\1. Professionall\Data Science\09-04-2023\5. RANDOM FOREST\Social_Network_Ads.csv')

X = ds.iloc[:, [2,3]].values
y = ds.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=400, criterion='log_loss')
RF.fit(X_train, y_train)

y_pred = RF.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
ac

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr

bias = RF.score(X_train, y_train)
bias

variance = RF.score(X_test, y_test)
variance