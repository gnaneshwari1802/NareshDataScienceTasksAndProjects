#!/usr/bin/env python
# coding: utf-8

# Business Objective - 
# 
# - The case study aims to identify defaulters, which indicate if a client has difficulty paying their installments which may be used for taking actions such as denying the loan, reducing the amount of   loan, lending (to risky applicants).
# - As we need to predict weather defaulter or not. that is this problem statement comes under
#   classification problem. 
# - As per the use case 5% defaulters and 95% are non-defaulters that means this kind of dataset is  highly imbalaced dataset

# A - Below approches need to follow for Data preprocessing & Data cleaing steps -
# -------------------------------------------------------------------------------
# 1> Importing all the requeired libraries 
# 2> Importing dataset
# 3> Read the dataframe and try to understand the length, columns , datatypes, we should understand
#    and we have to check any missing values are availabe or not
# 4> Data cleaning steps 
# 	- we will check about missing values & we will check how much % of missing values
# 	- if numerical data is missing then we have to apply mean strategy
# 	- if categorical value is missing then use dummy variable to replace and convert with numerical value
# 5> Analyze and delete unnecesary columns
# 6> Feature engineering is required to do that means try to find out the corelation between variables using heatmap visualization 
# 7> Data type conversion (all the object datatype will covert from object data type to int,float or string datatype based on the requirment

# In[6]:


# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 
#this will ignore the warnings.it wont␣ ↪display warnings in notebook
get_ipython().run_line_magic('pylab', 'inline')
from io import StringIO


# In[9]:


g='''a,b,c,1,2,0,g'''
g=unicode(g)
y=pd.read_csv(StringIO(g))
y


# In[5]:


df=pd.read_csv(r"C:\Users\M GNANESHWARI\Desktop\application_data.csv")


# In[54]:


df


# In[ ]:


df.values


# In[4]:


len(df)


# In[5]:


df.describe


# In[6]:


df.describe()


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.info()


# In[10]:


df.columns


# In[52]:


type(df['SK_ID_CURR'])


# In[13]:


df.isna


# In[14]:


df.isna()


# In[16]:


df.isna().count()


# In[8]:


df[df.isna().any(axis=1)]


# In[17]:


df.isna().sum().count()


# In[18]:


df.isna().sum().sum()


# In[81]:


df.isna().sum()


# In[91]:


df.columns[df.isnull().any()]


# In[90]:


nan_cols = [i for i in df.columns if df[i].isnull().any()]
nan_cols


# In[85]:


df.isna().sum().any()


# In[86]:


df.isna()


# In[87]:


df.isna().any()


# In[79]:


df.columns.isna().sum()


# In[88]:


df.columns[df.isna().any()].tolist()


# In[82]:


df.isnull().values.any()


# In[ ]:


df.fillna(df.mean())


# In[67]:


df['AMT_REQ_CREDIT_BUREAU_DAY']=df['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(np.mean(pd.to_numeric(df['AMT_REQ_CREDIT_BUREAU_DAY'])))


# In[68]:


df['AMT_REQ_CREDIT_BUREAU_WEEK']=df['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(np.mean(pd.to_numeric(df['AMT_REQ_CREDIT_BUREAU_WEEK'])))


# In[69]:


df['AMT_REQ_CREDIT_BUREAU_MON']=df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(np.mean(pd.to_numeric(df['AMT_REQ_CREDIT_BUREAU_MON'])))


# In[70]:


df['AMT_REQ_CREDIT_BUREAU_QRT']=df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(np.mean(pd.to_numeric(df['AMT_REQ_CREDIT_BUREAU_QRT'])))


# In[71]:


df['AMT_REQ_CREDIT_BUREAU_YEAR']=df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(np.mean(pd.to_numeric(df['AMT_REQ_CREDIT_BUREAU_YEAR'])))


# In[72]:


df


# In[80]:


df.isna().sum().sum()


# In[59]:


df.isna().sum().sum()/df.size


# In[4]:


(df.isna().sum().sum()*100)/df.size


# In[21]:


df.dtypes


# In[24]:


df.drop()


# In[25]:


len(df.columns)


# In[27]:


df=df.str.replace(r'\W','')


# In[28]:


df.index


# In[30]:


df.shape


# In[32]:


ff=df.dropna()


# In[ ]:


df[df.isnull]=df[df.isnull].fillna(df[df.isnull].mode())


# In[ ]:


df[df.isnull]=df[df.isnull].fillna(df[df.isnull].mean())


# In[ ]:


# Code to get number of categories in missing value columns
print("Number of Categories in: ")
for ColName in df[['Embarked','Cabin_Serial','Cabin']]:
    print("{} = {}".format(ColName,       len(df[ColName].unique())))


# In[33]:


df


# In[34]:


df.isnull().any().any()


# In[35]:


df.shape


# In[36]:


df.size


# In[37]:


df.isna


# In[38]:


df.isna()


# In[39]:


df.isna().count()


# In[63]:


df.fillna(0,inplace=True)


# In[64]:


df.isnull().values.any()


# In[62]:


df.isnull().sum()


# In[40]:


df.isna().count().sum()


# In[41]:


df.isna().sum()


# In[42]:


df.sum()


# In[43]:


df


# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')

df.hist(column='SK_ID_CURR', figsize=(10,10))


# In[50]:


df.boxplot(column='SK_ID_CURR', figsize=(10,10))


# In[51]:


df['NAME_CONTRACT_TYPE'].head()


# In[76]:


df


# In[ ]:




