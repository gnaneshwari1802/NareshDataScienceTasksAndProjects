#!/usr/bin/env python
# coding: utf-8

# Import libraries

# In[4]:


# This Python 3 environment comes with many helpful analytics libraries␣ ↪installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[3]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list␣ ↪all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# /kaggle/input/heart-disease-uci/heart.csv
# We can see that the input folder contains one input file named heart.csv.

# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")


# In[6]:


# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# I have imported the libraries. The next step is to import the datasets.
# 1.6 5. Import dataset
# Back to Table of Contents
# I will import the dataset with the usual pandas read_csv() function which is used to import CSV
# (Comma Separated Value) files.

# In[6]:


df = pd.read_csv('D:\DataScience\heart-disease-dataset\heart.csv')


#  Exploratory Data Analysis
#     Check shape of the dataset
# • It is a good idea to first check the shape of the dataset

# In[8]:


# print the shape
print('The shape of the dataset : ', df.shape)


# Preview the dataset

# In[9]:


# preview dataset
df.head()


# Summary of dataset

# In[10]:


# summary of dataset
df.info()


# Dataset description

# • The dataset contains several columns which are as follows -
# – age : age in years
# – sex : (1 = male; 0 = female)
# – cp : chest pain type
# – trestbps : resting blood pressure (in mm Hg on admission to the hospital)
# – chol : serum cholestoral in mg/dl
# – fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# – restecg : resting electrocardiographic results
# – thalach : maximum heart rate achieved
# – exang : exercise induced angina (1 = yes; 0 = no)
# – oldpeak : ST depression induced by exercise relative to rest
# – slope : the slope of the peak exercise ST segment
# – ca : number of major vessels (0-3) colored by flourosopy
# – thal : 3 = normal; 6 = fixed defect; 7 = reversable defect
# – target : 1 or 0

# Check the data types of columns

# • The above df.info() command gives us the number of filled values along with the data types
# of columns.
# • If we simply want to check the data type of a particular column, we can use the following
# command

# In[11]:


df.dtypes


# Important points about dataset

# sex is a character variable. Its data type should be object. But it is encoded as (1 = male;
# 0 = female). So, its data type is given as int64.
# • Same is the case with several other variables - fbs, exang and target. 6
# • fbs (fasting blood sugar) should be a character variable as it contains only 0 and 1 as
# values (1 = true; 0 = false). As it contains only 0 and 1 as values, so its data type is given
# as int64.
# • exang (exercise induced angina) should also be a character variable as it contains only
# 0 and 1 as values (1 = yes; 0 = no). It also contains only 0 and 1 as values, so its data type
# is given as int64.
# • target should also be a character variable. But, it also contains 0 and 1 as values. So, its
# data type is given as int64.

# Statistical properties of dataset

# In[12]:


# statistical properties of dataset
df.describe()


# Important points to note
# • The above command df.describe() helps us to view the statistical properties of numerical
# 7
# variables. It excludes character variables.
# • If we want to view the statistical properties of character variables, we should run the following
# command -
# df.describe(include=['object'])
# • If we want to view the statistical properties of all the variables, we should run the following
# command -
# df.describe(include='all')

# In[13]:


df.describe(include=['object'])


# If we want to view the statistical properties of all the variables, we should run the following
# command -

# In[14]:


df.describe(include='all')


# View column names

# In[15]:


df.columns


#  Univariate analysis

# Analysis of target feature variable

# • Our feature variable of interest is target. • It refers to the presence of heart disease in the patient.
# • It is integer valued as it contains two integers 0 and 1 - (0 stands for absence of heart disease
# and 1 for presence of heart disease).
# • So, in this section, I will analyze the target variable.

# Check the number of unique values in target variable

# In[16]:


df['target'].nunique()


# We can see that there are 2 unique values in the target variable.

# View the unique values in target variable

# In[17]:


df['target'].unique()


# Comment So, the unique values are 1 and 0. (1 stands for presence of heart disease and 0 for
# absence of hear disease).
# Frequency distribution of target variable

# In[18]:


df['target'].value_counts()


# Comment
# • 1 stands for presence of heart disease. So, there are 165 patients suffering from heart disease.
# • Similarly, 0 stands for absence of heart disease. So, there are 138 patients who do not have
# any heart disease.
# • We can visualize this information below.

# Visualize frequency distribution of target variable

# In[19]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", data=df)
plt.show()


# Interpretation

# • The above plot confirms the findings that -
# – There are 165 patients suffering from heart disease, and
# – There are 138 patients who do not have any heart disease.

# Frequency distribution of target variable wrt sex

# In[21]:


df.groupby('sex')['target'].value_counts()


# #### Comment
# • sex variable contains two integer values 1 and 0 : (1 = male; 0 = female).
# • target variable also contains two integer values 1 and 0 : (1 = Presence of heart disease; 0
# = Absence of heart disease)

# We can visualize the value counts of the sex variable wrt target as follows -

# In[22]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="sex", hue="target", data=df)
plt.show()


# Comment
# • The above plot segregate the values of target variable and plot on two different columns
# labelled as (sex = 0, sex = 1).
# • I think it is more convinient way of interpret the plots.

# We can plot the bars horizontally as follows :

# In[23]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(y="target", hue="sex", data=df)
plt.show()


# We can use a different color palette as follows :

# In[1]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", data=df, palette="Set3")
plt.show()


# In[4]:


"""The countplot is used to represent the occurrence(counts) of the observation present in the categorical variable. 
It uses the concept of a bar chart for the visual depiction.
Parameters-
The following parameters are specified when we create a countplot, let us get a brief idea of them-

x and y- This parameter specifies the data we refer to for representation and then observes the highlighted patterns.
color- This parameter specifies the color that can give a good appearance to our plot.
palette- It takes the value of the palette. It is mostly used to show the hue variable.
hue- This parameter specifies the column name.
data- This parameter specifies the data frame we would like to take for the representation. For instance, data can be an array.
dodge- This parameter is an optional one and it accepts a Boolean value as input.
saturation- This parameter accepts a float value. A variation in the intensity of colors can be observed when we specify this.
hue_order- The parameter hue_order takes strings as an input.
kwargs- The parameter kwargs specifies the key and value mappings.
ax- The parameter ax is an optional one and is used to take axes on which plots are created.
orient- The parameter orient is an optional one and tells the orientation of the plot that we need, horizontal or vertical.
"""


# We can use plt.bar keyword arguments for a different look :

# In[25]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", data=df, facecolor=(0, 0, 0, 0), linewidth=5,edgecolor=sns.color_palette("dark", 3))
plt.show()


# Comment
# • I have visualize the target values distribution wrt sex. • We can follow the same principles and visualize the target values distribution wrt fbs
# (fasting blood sugar) and exang (exercise induced angina).

# In[26]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", hue="fbs", data=df)
plt.show()


# In[27]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="target", hue="exang", data=df)
plt.show()


# Findings of Univariate Analysis
# Findings of univariate analysis are as follows:-
# • Our feature variable of interest is target. • It refers to the presence of heart disease in the patient.
# • It is integer valued as it contains two integers 0 and 1 - (0 stands for absence of heart disease
# and 1 for presence of heart disease).

# Bivariate Analysis
# 

# Estimate correlation coefficients

# Our dataset is very small. So, I will compute the standard correlation coefficient (also called
# Pearson’s r) between every pair of attributes. I will compute it using the df.corr() method as
# follows:-

# In[54]:


correlation = df.corr()


# The target variable is target. So, we should check how each attribute correlates with the target
# variable. We can do it as follows:-

# In[55]:


correlation['target'].sort_values(ascending=False)


# Interpretation of correlation coefficient

# • The correlation coefficient ranges from -1 to +1.
# • When it is close to +1, this signifies that there is a strong positive correlation. So, we can
# see that there is no variable which has strong positive correlation with target variable.
# • When it is clsoe to -1, it means that there is a strong negative correlation. So, we can see
# that there is no variable which has strong negative correlation with target variable.
# • When it is close to 0, it means that there is no correlation. So, there is no correlation between
# target and fbs. • We can see that the cp and thalach variables are mildly positively correlated with target
# variable. So, I will analyze the interaction between these features and target variable.

# Analysis of target and cp variable

# Explore cp variable

# • cp stands for chest pain type.
# • First, I will check number of unique values in cp variable.

# In[30]:


df['cp'].nunique()


# So, there are 4 unique values in cp variable. Hence, it is a categorical variable.
# Now, I will view its frequency distribution as follows :

# In[31]:


df['cp'].value_counts()


# Comment
# • It can be seen that cp is a categorical variable and it contains 4 types of values - 0, 1, 2 and
# 3.

# Visualize the frequency distribution of cp variable

# In[32]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="cp", data=df)
plt.show()


# Frequency distribution of target variable wrt cp

# In[7]:


df.groupby('cp')['target'].value_counts()


# Comment
# • cp variable contains four integer values 0, 1, 2 and 3.
# • target variable contains two integer values 1 and 0 : (1 = Presence of heart disease; 0 =
# Absence of heart disease)
# 20
# • So, the above analysis gives target variable values categorized into presence and absence of
# heart disease and groupby cp variable values.

# We can visualize this information below.

# We can visualize the value counts of the cp variable wrt target as follows -

# In[8]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x="cp", hue="target", data=df)
plt.show()


# Interpretation

# We can see that the values of target variable are plotted wrt cp. • target variable contains two integer values 1 and 0 : (1 = Presence of heart disease; 0 =
# Absence of heart disease)
# • The above plot confirms our above findings,

# Alternatively, we can visualize the same information as follows :

# In[11]:


ax = sns.catplot(x="target", col="cp", data=df, kind="count", height=8,aspect=1)


# Analysis of target and thalach variable

# Explore thalach variable

# • thalach stands for maximum heart rate achieved.
# • I will check number of unique values in thalach variable as follows :

# In[12]:


df['thalach'].nunique()


# • So, number of unique values in thalach variable is 91. Hence, it is numerical variable.
# • I will visualize its frequency distribution of values as follows :

# Visualize the frequency distribution of thalach variable

# In[13]:


f, ax = plt.subplots(figsize=(10,6))
x = df['thalach']
ax = sns.distplot(x, bins=10)
plt.show()


# Comment
# • We can see that the thalach variable is slightly negatively skewed.

# We can use Pandas series object to get an informative axis label as follows :

# In[16]:


f, ax = plt.subplots(figsize=(10,6))
x = df['thalach']
x = pd.Series(x, name="thalach variable")
ax = sns.distplot(x, bins=10)
plt.show()


# We can plot the distribution on the vertical axis as follows:-

# In[17]:


f, ax = plt.subplots(figsize=(10,6))
x = df['thalach']
ax = sns.distplot(x, bins=10, vertical=True)
plt.show()


# Seaborn Kernel Density Estimation (KDE) Plot

# • The kernel density estimate (KDE) plot is a useful tool for plotting the shape of a distribution.
# • The KDE plot plots the density of observations on one axis with height along the other axis.

# • We can plot a KDE plot as follows :

# In[19]:


f, ax = plt.subplots(figsize=(10,6))
x = df['thalach'] 
x = pd.Series(x, name="thalach variable")
ax = sns.kdeplot(x)
plt.show()


# We can shade under the density curve and use a different color as follows:

# In[21]:


f, ax = plt.subplots(figsize=(10,6))
x = df['thalach'] 
x = pd.Series(x, name="thalach variable")
ax = sns.kdeplot(x, shade=True, color='r')
plt.show()


# Histogram

# • A histogram represents the distribution of data by forming bins along the range of the data
# and then drawing bars to show the number of observations that fall in each bin.
# • We can plot a histogram as follows :

# In[22]:


f, ax = plt.subplots(figsize=(10,6))
x = df['thalach']
ax = sns.distplot(x, kde=False, rug=True, bins=10)
plt.show()


# Visualize frequency distribution of thalach variable wrt target

# In[23]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="thalach", data=df)
plt.show()


# Interpretation

# • We can see that those people suffering from heart disease (target = 1) have relatively higher
# heart rate (thalach) as compared to people who are not suffering from heart disease (target
# = 0).

# We can add jitter to bring out the distribution of values as follows

# In[24]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="thalach", data=df, jitter = 0.01)
plt.show()


# Visualize distribution of thalach variable wrt target with boxplot

# In[25]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="target", y="thalach", data=df)
plt.show()


# Interpretation The above boxplot confirms our finding that people suffering from heart disease
# (target = 1) have relatively higher heart rate (thalach) as compared to people who are not suffering
# from heart disease (target = 0).

#  Findings of Bivariate Analysis
# Findings of Bivariate Analysis are as follows –
# • There is no variable which has strong positive correlation with target variable.
# • There is no variable which has strong negative correlation with target variable.
# • There is no correlation between target and fbs. • The cp and thalach variables are mildly positively correlated with target variable.
# • We can see that the thalach variable is slightly negatively skewed.
# • The people suffering from heart disease (target = 1) have relatively higher heart rate (thalach)
# as compared to people who are not suffering from heart disease (target = 0).
# • The people suffering from heart disease (target = 1) have relatively higher heart rate (thalach)
# as compared to people who are not suffering from heart disease (target = 0).

# Multivariate analysis

# • The objective of the multivariate analysis is to discover patterns and relationships in the
# dataset.

# Discover patterns and relationships

# • An important step in EDA is to discover patterns and relationships between variables in the
# dataset.
# • I will use heat map and pair plot to discover the patterns and relationships in the dataset.
# • First of all, I will draw a heat map.

# Heat Map

# In[56]:


plt.figure(figsize=(16,12))
plt.title('Correlation Heatmap of Heart Disease Dataset') 
a=sns.heatmap(correlation, square=True, annot=True, fmt='.2f',linecolor='white') 
a.set_xticklabels(a.get_xticklabels(), rotation=90) 
a.set_yticklabels(a.get_yticklabels(), rotation=30)
plt.show()


# Box-plot of chol variable

# . Check with ASSERT statement
# Back to Table of Contents
# • We must confirm that our dataset has no missing values.
# • We can write an assert statement to verify this.
# • We can use an assert statement to programmatically check that no missing, unexpected 0 or
# negative values are present.
# • This gives us confidence that our code is running properly.
# • Assert statement will return nothing if the value being tested is true and will throw an
# AssertionError if the value is false.
# • Asserts
# – assert 1 == 1 (return Nothing if the value is True)
# – assert 1 == 2 (return AssertionError if the value is False)

# Analyze chol and thalach variable

# 

#  Pair Plot

# In[52]:


num_var = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target' ]
sns.pairplot(df[num_var], kind='scatter', diag_kind='hist')
plt.show()


# Comment
# • I have defined a variable num_var. Here age, trestbps, chol`, `thalach` and `oldpeak
# are numerical variables and target is the categorical variable.
# • So, I wll check relationships between these variables.

# Analysis of age and other variables

# Check the number of unique values in age variable

# In[51]:


df['age'].nunique()


# View statistical summary of age variable

# In[ ]:


df['age'].describe()


# Interpretation

# In[50]:


f, ax = plt.subplots(figsize=(10,6))
x = df['age']
ax = sns.distplot(x, bins=10)
plt.show()


#  Analyze age and target variable

# Visualize frequency distribution of age variable wrt target

# In[49]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="target", y="age", data=df)
plt.show()


# Visualize distribution of age variable wrt target with boxplot

# In[48]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="target", y="age", data=df)
plt.show()


# Analyze age and trestbps variable

# In[47]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.scatterplot(x="age", y="trestbps", data=df)
plt.show()


# Interpretation

# In[ ]:





# In[46]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="age", y="trestbps", data=df)
plt.show()


# Analyze age and chol variable

# In[ ]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.scatterplot(x="age", y="chol", data=df)
plt.show()


# In[45]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="age", y="chol", data=df)
plt.show()


# In[44]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.scatterplot(x="chol", y = "thalach", data=df)
plt.show()


# In[43]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="chol", y="thalach", data=df)
plt.show()


# Useful commands to detect missing values
# • df.isnull()
# The above command checks whether each cell in a dataframe contains missing values or not. If the
# cell contains missing value, it returns True otherwise it returns False.
# • df.isnull().sum()
# The above command returns total number of missing values in each column in the dataframe.
# • df.isnull().sum().sum()
# It returns total number of missing values in the dataframe.
# • df.isnull().mean()
# It returns percentage of missing values in each column in the dataframe.
# • df.isnull().any()
# It checks which column has null values and which has not. The columns which has null values
# returns TRUE and FALSE otherwise.
# • df.isnull().any().any()
# It returns a boolean value indicating whether the dataframe has missing values or not. If dataframe
# contains missing values it returns TRUE and FALSE otherwise.
# • df.isnull().values.any()
# It checks whether a particular column has missing values or not. If the column contains missing
# values, then it returns TRUE otherwise FALSE.
# • df.isnull().values.sum()
# It returns the total number of missing values in the dataframe.

# In[42]:


# check for missing values
df.isnull().sum()


# In[41]:


#assert thaat there are no missing values in the dataframe
assert pd.notnull(df).all().all()


# In[40]:


#assert all values are greater than or equal to 0
assert (df >= 0).all().all()


#  age variable

# In[39]:


df['age'].describe()


# Box-plot of age variable

# In[38]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["age"])
plt.show()


# trestbps variable

# In[37]:


df['trestbps'].describe()


# Box-plot of trestbps variable

# In[36]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["trestbps"])
plt.show()


# chol variable

# In[35]:


df['chol'].describe()


# In[34]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["chol"])
plt.show()


# thalach variable

# In[32]:


df['thalach'].describe()


# Box-plot of thalach variable

# In[31]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["thalach"])
plt.show()


# oldpeak variable

# In[30]:


df['oldpeak'].describe()


# Box-plot of oldpeak variable

# In[ ]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df["oldpeak"])
plt.show()

