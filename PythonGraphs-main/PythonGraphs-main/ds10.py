# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:42:17 2019

@author: M GNANESHWARI
"""

import pandas as p
data_cars=p.read_csv("D:\\DataScience\\titanic-dataset-from-kaggle\\test.csv")
#da_cars=p.DataFrame(data_cars)
print(data_cars)
for i in range(0,len(data_cars["PassengerId"]),1):
    if(data_cars["PassengerId"][i]<=1309):
        data_cars["Cabin"][i]="NaN"
    else:
        data_cars["Cabin"][i]!="NaN"
print(data_cars["Cabin"].value_counts())
print(data_cars.columns)        
print(data_cars.index)
bimas_tab=p.crosstab(index=data_cars["Pclass"],columns="count")
print(pclass_tab)

