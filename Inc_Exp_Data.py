#!/usr/bin/env python
# coding: utf-8

# Descrptive Statistics Analysis and Visualisation

# In[61]:


import matplotlib.pyplot as plt 
import matplotlib.style as style
from collections import Counter 
The collection Module in Python provides different types of containers. A Container is an object that is used to store different objects and provide a way to access the contained objects and iterate over them. Some of the built-in containers are Tuple, List, Dictionary, etc. In this article, we will discuss the different containers provided by the collections module.
get_ipython().run_line_magic('matplotlib', 'inline')
What is Run_line_magic?
magic() method basically does the of prepending the argument provided with a % if you were in an ipython shell. Third, we use the run_line_magic() method on the ip object to run our line magic. This method takes two arguments: the name of the magic function and the remaining of the arguments for that magic.
import pandas as pd
import numpy as np
import itertools
import seaborn as sns


# In[24]:


import warnings 
warnings.filterwarnings("ignore") 


# In[5]:


amir=pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\Inc_Exp_Data.csv')


# In[6]:


amir.head() 


# In[7]:


amir.tail() 


# In[56]:


amir["Highest_Qualified_Member"].value_counts().plot(kind="bar") 


# In[55]:


amir["No_of_Earning_Members"].value_counts().plot(kind="bar") 


# In[53]:


f, ax=plt.subplots(figsize=(8,6))
ax=sns.lineplot(x="Annual_HH_Income",y="Emi_or_Rent_Amt",data=amir)
plt.show()


# In[54]:


f, ax=plt.subplots(figsize=(8,6))
x=amir["Annual_HH_Income"]
y=pd.Series(x,name="Annual_HH_Income")
ax=sns.kdeplot(x,shade=True,color="r")
What is a KDE plot in Seaborn?
Kdeplot is a Kernel Distribution Estimation Plot which depicts the probability density function of the continuous or non-parametric data variables i.e. we can plot for the univariate or multiple variables altogether. Using the Python Seaborn module, we can build the Kdeplot with various functionality added to it.
plt.show()


# In[8]:


amir.info() 


# In[60]:


amir.describe(include="object")


# In[59]:


amir.describe()


# In[9]:


amir.describe().T 


# In[10]:


amir.shape 


# In[62]:


amir.isnull().any()


# In[63]:


amir.isnull().sum()


# In[11]:


amir.isna().any() 


# In[12]:


amir["Mthly_HH_Expense"].mean() 


# In[13]:


amir["Mthly_HH_Expense"].median() 


# In[14]:


amir["Mthly_HH_Expense"].mode() 


# In[15]:


"""Quantiles are points in a distribution that relate to the rank order of values in that distribution. For a sample, you can find any quantile by sorting the sample. The middle value of the sorted sample (middle quantile, 50th percentile) is known as the median. The limits are the minimum and maximum values."""


# In[75]:


monthly_expences = pd.crosstab(index=amir["Mthly_HH_Expense"],columns="count") 
What is PD crosstab?
The Crosstab function is one of the many methods that help you reshape your data in Pandas. At first glance, its purpose appears to be comparable to pivot, and you can perform many of the same things with Pandas Crosstab as you can with Pandas Pivot Table.
monthly_expences.reset_index(inplace=True)
  What does reset_index do in Pandas?
Definition and Usage. The reset_index() method allows you reset the index back to the default 0, 1, 2 etc indexes. By default this method will keep the "old" idexes in a column named "index", to avoid this, use the drop parameter.
monthly_expences[monthly_expences["count"]== amir.Mthly_HH_Expense.value_counts()] 
value_counts() The value_counts() function returns a Series that contain counts of unique values. It returns an object that will be in descending order so that its first element will be the most frequently-occurred element. By default, it excludes NA values.

# In[26]:


amir["Mthly_HH_Income"].quantile([0.25,0.75]) 


# In[27]:


23550.0 -(1.5*26825.0) 


# In[20]:


amir.Mthly_HH_Expense[18]


# In[18]:


monthly_exp_tem = pd.crosstab(index =amir['Mthly_HH_Expense'],columns = 'count')


# In[19]:


monthly_exp_tem.reset_index(inplace=True)


# In[21]:


monthly_exp_tem[monthly_exp_tem['count'] == amir.Mthly_HH_Expense.value_counts().max()]


# In[81]:


vis2=sns.countplot(data=amir,x='No_of_Earning_Members')
plt.show()


# In[80]:


vis=sns.countplot(data=amir,x='Highest_Qualified_Member')


# In[25]:


plt.figure(figsize= (10,4))
sns.countplot(amir['Highest_Qualified_Member'])
plt.show()


# In[26]:


amir['Highest_Qualified_Member'].value_counts()


# calculate IQR

# In[50]:


amir.plot(x='Mthly_HH_Income', y='Mthly_HH_Expense')

q1=amir["Mthly_HH_Expense"].quantile(0.75)

q3=amir["Mthly_HH_Expense"].quantile(0.25)
IQR=q1-q3


# In[49]:


amir.plot(x='Mthly_HH_Income', y='Mthly_HH_Expense')

q1=amir["Mthly_HH_Expense"].quantile(0.75)

q3=amir["Mthly_HH_Expense"].quantile(0.25)


# In[47]:


amir.plot(x='Mthly_HH_Income', y='Mthly_HH_Expense')


# In[58]:


import matplotlib.pyplot as plt
plt.boxplot(amir["Mthly_HH_Income"])
plt.show()


# In[27]:


amir.plot(x='Mthly_HH_Income', y='Mthly_HH_Expense')
plt.show()


# In[29]:


IQR = amir['Mthly_HH_Expense'].quantile(0.75)-amir['Mthly_HH_Expense'].quantile(0.25)
IQR


# In[30]:


IQR = amir['Mthly_HH_Expense'].quantile(0.50)
IQR


# In[31]:


pd.DataFrame(amir.iloc[:,:].std().to_frame('standard deviation'))


# The iloc function in Python returns a view of the selected rows and columns from a Pandas DataFrame. This view can be used to access, modify, or delete the selected data. The returned view is a Pandas DataFrame or Series, depending on the number of rows or columns selected.

# In[32]:


pd.DataFrame(amir.iloc[:,:].var().to_frame('variance'))


# The to_frame() function is used to convert Series to DataFrame. The passed name should substitute for the series name (if it has one). DataFrame representation of Series.

# In[33]:


amir['Highest_Qualified_Member'].value_counts().to_frame()


# In[35]:


amir['Highest_Qualified_Member'].value_counts()


# In[36]:


amir['Highest_Qualified_Member'].().plot(kind='bar')
plt.show()


# STANDARD DEVIATION 

# In[45]:


amir.head(1)


# In[46]:


vis=sns.lmplot(data=amir,x="Mthly_HH_Income",y="Mthly_HH_Expense",fit_reg=False)


# In[52]:


f,ax=plt.subplots(figsize=(8,6))
sns.violinplot(x=amir["Mthly_HH_Expense"])
plt.show()


# In[42]:


z=sns.violinplot(data=amir,x="No_of_Fly_Members",b="No_of_Earning_Members")


# In[41]:


pd.DataFrame(amir.iloc[:,0:5].std().to_frame()).T 


# First three column variance

# In[38]:


pd.DataFrame(amir.iloc[:,0:4].var().to_frame()).T 


# In[39]:


pd.DataFrame(amir.iloc[:,0:4].var().to_frame())

What is To_frame () in Python?
to_frame() function is used to convert the given series object to a dataframe. Syntax: Series.to_frame(name=None) Parameter : name : The passed name should substitute for the series name (if it has one).
# In[37]:


amir["Highest_Qualified_Member"].value_counts().to_frame().T 


# In[79]:


pd.DataFrame(amir.iloc[:,0:5].var().to_frame()).T


# In[ ]:




