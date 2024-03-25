# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:07:33 2019

@author: M GNANESHWARI
"""
"""
import pandas as pd
data=pd.read_csv("D:\\Datascience\\lendingdata.csv")
print(sum(data.isnull().sum()))
data.dropna(inplace=True)
print(sum(data.isnull().sum()))
pd.get_dummies(data)
cols=data.columns.values
print(cols)
x=data.drop(['status'],axis=1)
y=data['status']
cols=data1.columns.values
x1=list[cols]-set(['statues'])
"""
import pandas as pd
data=pd.read_csv("D:\\Datascience\\microlending_data.csv")
print(sum(data.isnull().sum()))
print(data.dtypes)
print(data.get_dtype_counts())
#.dtypes.value_counts()
"""
#print(data['borrower_genders'].fillna(data['borrower_genders'].median()[0]))
data.dropna(axis=0,inplace=True)
print(data)
print(data.sum(axis=0,skipna=True))
new=data["borrower_genders"] and data["]
"""
s=data.describe()
print(s)
"""
s=data.describe(include='0')
print(s)
"""
print(data['loan_amount'].value_counts())
c=data.corr()
print(c)
print(data.columns)
g=pd.crosstab(index=data["loan_amount"],
              columns='count',
              normalize=True)
print(g)
g1=pd.crosstab(index=data["borrower_genders"],
              columns='count',
              normalize=True)    
print(g1) 
g2=pd.crosstab(index=data["loan_amount"],
               columns=data['sector'],
               )         
print(g2)


def getDuplicateColumns(data):
    '''
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(data.shape[1]):
        # Select column at xth index.
        col = data.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, data.shape[1]):
            # Select column at yth index.
            otherCol = data.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(data.columns.values[y])
 
    return list(duplicateColumnNames)


# Get list of duplicate columns
duplicateColumnNames = getDuplicateColumns(data)
 
print('Duplicate Columns are as follows')
for col in duplicateColumnNames:
    print('Column name : ', col)
newDf = data.drop(columns=getDuplicateColumns(data))
 
print("Modified Dataframe", newDf, sep='\n')
 
