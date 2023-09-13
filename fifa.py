#!/usr/bin/env python
# coding: utf-8

#  Seaborn tutorial for beginners

#  Import libraries

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries␣ ↪installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(style="whitegrid")
"""
The next line sns.set() will load seaborn's default theme and color palette to the session. Run the code below and watch the change in the chart area and the text.
"""
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list␣ ↪all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# In[3]:


# ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Read dataset

# In[10]:


fifa19 = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\fifa.csv', index_col=0)


#  Exploratory Data Analysis
# Preview the dataset

# In[11]:


fifa19.head()


# View summary of dataset

# In[12]:


fifa19.info()


# In[13]:


fifa19['Body Type'].value_counts()


# Explore Age variable
# Visualize distribution of Age variable with Seaborn distplot() function

# In[14]:


f, ax = plt.subplots(figsize=(8,6))
x = fifa19['Age']
ax = sns.distplot(x, bins=10)
plt.show()


#  Comment

# • It can be seen that the Age variable is slightly positively skewed.
# We can use Pandas series object to get an informative axis label as follows-

# In[16]:


f, ax = plt.subplots(figsize=(8,6))
x = fifa19['Age'] 
x = pd.Series(x, name="Age variable")
ax = sns.distplot(x, bins=10)
plt.show()


# We can plot the distribution on the vertical axis as follows:-
# 

# In[17]:


f, ax = plt.subplots(figsize=(8,6))
x = fifa19['Age']
ax = sns.distplot(x, bins=10, vertical = True)
plt.show()

"""
What is a Seaborn Distplot?
A Distplot or distribution plot, depicts the variation in the data distribution. Seaborn Distplot represents the overall distribution of continuous data variables.

The Seaborn module along with the Matplotlib module is used to depict the distplot with different variations in it. The Distplot depicts the data by a histogram and a line in combination to it.

Creating a Seaborn Distplot
Python Seaborn module contains various functions to plot the data and depict the data variations. The seaborn.distplot() function is used to plot the distplot. The distplot represents the univariate distribution of data i.e. data distribution of a variable against the density distribution.


"""
# Seaborn Kernel Density Estimation (KDE) Plot
# • The kernel density estimate (KDE) plot is a useful tool for plotting the shape of a distribution.
# • Seaborn kdeplot is another seaborn plotting function that fits and plot a univariate or bivariate
# kernel density estimate.
# • Like the histogram, the KDE plots encode the density of observations on one axis with height
# along the other axis.
# • We can plot a KDE plot as follows-

# In[18]:


f, ax = plt.subplots(figsize=(8,6))
x = fifa19['Age']
x = pd.Series(x, name="Age variable")
ax = sns.kdeplot(x)
plt.show()
"""
What is Kdeplot?
Kdeplot is a Kernel Distribution Estimation Plot which depicts the probability density function of the continuous or non-parametric data variables i.e. we can plot for the univariate or multiple variables altogether. Using the Python Seaborn module, we can build the Kdeplot with various functionality added to it.
Creating a Univariate Seaborn Kdeplot
The seaborn.kdeplot() function is used to plot the data against a single/univariate variable. It represents the probability distribution of the data values as the area under the plotted curve.

"""
"""
What Is the Probability Density Function?
A function that defines the relationship between a random variable and its probability, such that you can find the probability of the variable using the function, is called a Probability Density Function (PDF) in statistics.

The different types of variables. They are mainly of two types:

Discrete Variable: A variable that can only take on a certain finite value within a specific range is called a discrete variable. It usually separates the values by a finite interval, e.g., a sum of two dice. On rolling two dice and adding up the resulting outcome, the result can only belong to a set of numbers not exceeding 12 (as the maximum result of a dice throw is 6). The values are also definite.
Continuous Variable: A continuous random variable can take on infinite different values within a range of values, e.g., amount of rainfall occurring in a month. The rain observed can be 1.7cm, but the exact value is not known. It can, in actuality, be 1.701, 1.7687, etc. As such, you can only define the range of values it falls into. Within this value, it can take on infinite different values.
Now, consider a continuous random variable x, which has a probability density function, that defines the range of probabilities taken by this function as f(x). After plotting the pdf, you get a graph as shown below:                     

Probability_Density_Function_1.

Figure 1: Probability Density Function

In the above graph, you get a bell-shaped curve after plotting the function against the variable. The blue curve shows this. Now consider the probability of a point b. To find it, you need to find the area under the curve to the left of b. This is represented by P(b). To find the probability of a variable falling between points a and b, you need to find the area of the curve between a and b. As the probability cannot be more than P(b) and less than P(a), you can represent it as: 

P(a) <= X <= P(b).

Consider the graph below, which shows the rainfall distribution in a year in a city. The x-axis has the rainfall in inches, and the y-axis has the probability density function. The probability of some amount of rainfall is obtained by finding the area of the curve on the left of it.


"""

# We can shade under the density curve and use a different color as follows:-

# In[19]:


f, ax = plt.subplots(figsize=(8,6))
"""
The subplot() Function
The subplot() function takes three arguments that describes the layout of the figure.

The layout is organized in rows and columns, which are represented by the first and second argument.

The third argument represents the index of the current plot.

plt.subplot(1, 2, 1)
#the figure has 1 row, 2 columns, and this plot is the first plot.

plt.subplot(1, 2, 2)
#the figure has 1 row, 2 columns, and this plot is the second plot.
So, if we want a figure with 2 rows an 1 column (meaning that the two plots will be displayed on top of each other instead of side-by-side), we can write the syntax like this:

Creating multiple subplots using plt.subplots
pyplot.subplots creates a figure and a grid of subplots with a single call, while providing reasonable control over how the individual plots are created. For more advanced use cases you can use GridSpec for a more general subplot layout or Figure.add_subplot for adding subplots at arbitrary locations within the figure.

"""
x = fifa19['Age'] 
x = pd.Series(x, name="Age variable")
"""
What is a Series?
A Pandas Series is like a column in a table.

It is a one-dimensional array holding data of any type.
"""
ax = sns.kdeplot(x, shade=True, color='r')
plt.show()


# Histograms

# • A histogram represents the distribution of data by forming bins along the range of the data
# and then drawing bars to show the number of observations that fall in each bin.
# • A hist() function already exists in matplotlib.
# • We can use Seaborn to plot a histogram.

# In[20]:


f, ax = plt.subplots(figsize=(8,6))
x = fifa19['Age']
ax = sns.distplot(x, kde=False, rug=True, bins=10)
plt.show()


# We can plot a KDE plot alternatively as follows:-

# In[21]:


f, ax = plt.subplots(figsize=(8,6))
x = fifa19['Age']
ax = sns.distplot(x, hist=False, rug=True, bins=10)
plt.show()


# Explore Preferred Foot variable
# 
# Check number of unique values in Preferred Foot variable

# In[23]:


fifa19['Preferred Foot'].nunique()

"""
Definition and Usage
The nunique() method returns the number of unique values for each column.

By specifying the column axis (axis='columns'), the nunique() method searches column-wise and returns the number of unique values for each row.

Syntax
dataframe.nunique(axis, dropna)
Parameters
The parameters are keyword arguments.

Parameter	Value	Description
axis	0
1
'index'
'columns'	Optional, Which axis to check, default 0.
dropna	True
False	Optional, default True. Set to False if the result should NOT drop NULL values
Return Value
A Series with the number of unique values for each column or row.

This function does NOT make changes to the original DataFrame object.


"""
# We can see that there are two types of unique values in Preferred Foot variable.
# 
# Check frequency distribution of values in Preferred Foot variable

# In[25]:


fifa19['Preferred Foot'].value_counts()
"""
Pandas Series.value_counts()
The value_counts() function returns a Series that contain counts of unique values. It returns an object that will be in descending order so that its first element will be the most frequently-occurred element.

By default, it excludes NA values.

Syntax
Series.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)  
Parameters

normalize: If it is true, then the returned object will contain the relative frequencies of the unique values.
sort: It sort by the values.
ascending: It sort in the ascending order.
bins: Rather than counting the values, it groups them into the half-open bins that provide convenience for the pd.cut, which only works with numeric data.
dropna: It does not include counts of NaN.
Returns
It returns the counted series.
"""

# The Preferred Foot variable contains two types of values - Right and Left.

# Visualize distribution of values with Seaborn countplot() function.

# • A countplot shows the counts of observations in each categorical bin using bars.
# • It can be thought of as a histogram across a categorical, instead of quantitative, variable.
# • This function always treats one of the variables as categorical and draws data at ordinal
# positions (0, 1, … n) on the relevant axis, even when the data has a numeric or date type.
# 1. • We can visualize the distribution of values with Seaborn countplot() function as
# follows-

# In[26]:


f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="Preferred Foot", data=fifa19, color="c")
plt.show()
"""
seaborn.countplot() method is used to Show the counts of observations in each categorical bin using bars.
 

Syntax : seaborn.countplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, dodge=True, ax=None, **kwargs)
Parameters : This method is accepting the following parameters that are described below: 
 

x, y: This parameter take names of variables in data or vector data, optional, Inputs for plotting long-form data.
hue : (optional) This parameter take column name for colour encoding.
data : (optional) This parameter take DataFrame, array, or list of arrays, Dataset for plotting. If x and y are absent, this is interpreted as wide-form. Otherwise it is expected to be long-form.
order, hue_order : (optional) This parameter take lists of strings. Order to plot the categorical levels in, otherwise the levels are inferred from the data objects.
orient : (optional)This parameter take “v” | “h”, Orientation of the plot (vertical or horizontal). This is usually inferred from the dtype of the input variables but can be used to specify when the “categorical” variable is a numeric or when plotting wide-form data.
color : (optional) This parameter take matplotlib color, Color for all of the elements, or seed for a gradient palette.
palette : (optional) This parameter take palette name, list, or dict, Colors to use for the different levels of the hue variable. Should be something that can be interpreted by color_palette(), or a dictionary mapping hue levels to matplotlib colors.
saturation : (optional) This parameter take float value, Proportion of the original saturation to draw colors at. Large patches often look better with slightly desaturated colors, but set this to 1 if you want the plot colors to perfectly match the input color spec.
dodge : (optional) This parameter take bool value, When hue nesting is used, whether elements should be shifted along the categorical axis.
ax : (optional) This parameter take matplotlib Axes, Axes object to draw the plot onto, otherwise uses the current Axes.
kwargs : This parameter take key, value mappings, Other keyword arguments are passed through to matplotlib.axes.Axes.bar().
Returns: Returns the Axes object with the plot drawn onto it.

seaborn.countplot
seaborn.countplot(data=None, *, x=None, y=None, hue=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, width=0.8, dodge=True, ax=None, **kwargs)
Show the counts of observations in each categorical bin using bars.

A count plot can be thought of as a histogram across a categorical, instead of quantitative, variable. The basic API and options are identical to those for barplot(), so you can compare counts across nested variables.

Note that the newer histplot() function offers more functionality, although its default behavior is somewhat different.

Note

This function always treats one of the variables as categorical and draws data at ordinal positions (0, 1, … n) on the relevant axis, even when the data has a numeric or date type.


"""

# 
# We can show value counts for two categorical variables as follows-

# In[27]:


f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="Preferred Foot", hue="Real Face", data=fifa19)
plt.show()


# We can draw plot vertically as follows-

# In[28]:


f, ax = plt.subplots(figsize=(8, 6))
sns.countplot(y="Preferred Foot", data=fifa19, color="c")
plt.show()


#  Seaborn Catplot() function
# • We can use Seaborn Catplot() function to plot categorical scatterplots.
# • The default representation of the data in catplot() uses a scatterplot.
# • It helps to draw figure-level interface for drawing categorical plots onto a facetGrid.
# • This function provides access to several axes-level functions that show the relationship between a numerical and one or more categorical variables using one of several visual representations.
# • The kind parameter selects the underlying axes-level function to use.
# We can use the kind parameter to draw different plot kin to visualize the same data. We can use
# the Seaborn catplot() function to draw a countplot() as follows-

# In[29]:


g = sns.catplot(x="Preferred Foot", kind="count", palette="ch:.25", data=fifa19)


# Explore International Reputation variable
# 
# Check the number of unique values in International Reputation variable

# In[30]:


fifa19['International Reputation'].nunique()


# Check the distribution of values in International Reputation variable

# In[31]:


fifa19['International Reputation'].value_counts()


# Seaborn Stripplot() function
# • This function draws a scatterplot where one variable is categorical.
# • A strip plot can be drawn on its own, but it is also a good complement to a box or violin
# plot in cases where we want to show all observations along with some representation of the
# underlying distribution.
# • I will plot a stripplot with International Reputation as categorical variable and Potential
# as the other variable.

# In[32]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", data=fifa19)
plt.show()


# We can add jitter to bring out the distribution of values as follows-

# In[33]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", data=fifa19,jitter=0.01)
plt.show()


# We can nest the strips within a second categorical variable - Preferred Foot as folows-

# In[34]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", hue="Preferred Foot",
data=fifa19, jitter=0.2, palette="Set2", dodge=True)
plt.show()


# We can draw strips with large points and different aesthetics as follows-

# In[35]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", hue="Preferred Foot",
data=fifa19, palette="Set2", size=20, marker="D",
edgecolor="gray", alpha=.25)
plt.show()


#  Seaborn boxplot() function
# • This function draws a box plot to show distributions with respect to categories.
# • A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way
# that facilitates comparisons between variables or across levels of a categorical variable.
# • The box shows the quartiles of the dataset while the whiskers extend to show the rest of the
# distribution, except for points that are determined to be “outliers” using a method that is a
# function of the inter-quartile range.
# • I will plot the boxplot of the Potential variable as follows-
"""
In descriptive statistics, the interquartile range tells you the spread of the middle half of your distribution.

Quartiles segment any distribution that’s ordered from low to high into four equal parts. The interquartile range (IQR) contains the second and third quartiles, or the middle half of your data set.

Whereas the range gives you the spread of the whole data set, the interquartile range gives you the range of the middle half of a data set.
Calculate the interquartile range by hand
The interquartile range is found by subtracting the Q1 value from the Q3 value:

Formula	Explanation
Interquartile range formula	
IQR = interquartile range
Q3 = 3rd quartile or 75th percentile
Q1 = 1st quartile or 25th percentile
Q1 is the value below which 25 percent of the distribution lies, while Q3 is the value below which 75 percent of the distribution lies.

You can think of Q1 as the median of the first half and Q3 as the median of the second half of the distribution.

Methods for finding the interquartile range
Although there’s only one formula, there are various different methods for identifying the quartiles. You’ll get a different value for the interquartile range depending on the method you use.

Here, we’ll discuss two of the most commonly used methods. These methods differ based on how they use the median.

Exclusive method vs inclusive method
The exclusive method excludes the median when identifying Q1 and Q3, while the inclusive method includes the median in identifying the quartiles.

The procedure for finding the median is different depending on whether your data set is odd- or even-numbered.

When you have an odd number of data points, the median is the value in the middle of your data set. You can choose between the inclusive and exclusive method.
With an even number of data points, there are two values in the middle, so the median is their mean. It’s more common to use the exclusive method in this case.
While there is little consensus on the best method for finding the interquartile range, the exclusive interquartile range is always larger than the inclusive interquartile range.

The exclusive interquartile range may be more appropriate for large samples, while for small samples, the inclusive interquartile range may be more representative because it’s a narrower range.

Steps for the exclusive method
To see how the exclusive method works by hand, we’ll use two examples: one with an even number of data points, and one with an odd number.

Even-numbered data set
We’ll walk through four steps using a sample data set with 10 values.

Step 1: Order your values from low to high.
Ordered data set (even number)
Step 2: Locate the median, and then separate the values below it from the values above it.
With an even-numbered data set, the median is the mean of the two values in the middle, so you simply divide your data set into two halves.Data set in two halves
Step 3: Find Q1 and Q3.
Q1 is the median of the first half and Q3 is the median of the second half. Since each of these halves have an odd number of values, there is only one value in the middle of each half.
Finding Q1 and Q3
Step 4: Calculate the interquartile range.
Calculating the IQR
Odd-numbered data set
This time we’ll use a data set with 11 values.

Step 1: Order your values from low to high.
Ordered data set (odd number)
Step 2: Locate the median, and then separate the values below it from the values above it.
In an odd-numbered data set, the median is the number in the middle of the list. The median itself is excluded from both halves: one half contains all values below the median, and the other contains all the values above it.
Finding the median and dividing the data set into two halves
Step 3: Find Q1 and Q3.
Q1 is the median of the first half and Q3 is the median of the second half. Since each of these halves have an odd-numbered size, there is only one value in the middle of each half.
Finding Q1 and Q3 in an odd-numbered data set
Step 4: Calculate the interquartile range.
Calculating the IQR
"""
"""
Statistics - Quartiles and Percentiles
Quartiles and percentiles are measures of variation, which describes how spread out the data is.

Quartiles and percentiles are both types of quantiles.

Quartiles
Quartiles are values that separate the data into four equal parts.

Here is a histogram of the age of all 934 Nobel Prize winners up to the year 2020, showing the quartiles:

Histogram of the age of Nobel Prize winners with quartiles shown.

The quartiles (Q0,Q1,Q2,Q3,Q4) are the values that separate each quarter.

Between Q0 and Q1 are the 25% lowest values in the data. Between Q1 and Q2 are the next 25%. And so on.

Q0 is the smallest value in the data.
Q1 is the value separating the first quarter from the second quarter of the data.
Q2 is the middle value (median), separating the bottom from the top half.
Q3 is the value separating the third quarter from the fourth quarter
Q4 is the largest value in the data.
Calculating Quartiles with Programming
Quartiles can easily be found with many programming languages.

Using software and programming to calculate statistics is more common for bigger sets of data, as finding it manually becomes difficult.

Example
With Python use the NumPy library quantile() method to find the quartiles of the values 13, 21, 21, 40, 42, 48, 55, 72:

import numpy

values = [13,21,21,40,42,48,55,72]

x = numpy.quantile(values, [0,0.25,0.5,0.75,1])

print(x)

Percentiles
Percentiles are values that separate the data into 100 equal parts.

For example, The 95th percentile separates the lowest 95% of the values from the top 5%

The 25th percentile (P25%) is the same as the first quartile (Q1).

The 50th percentile (P50%) is the same as the second quartile (Q2) and the median.

The 75th percentile (P75%) is the same as the third quartile (Q3)

ADVERTISEMENT

Calculating Percentiles with Programming
Percentiles can easily be found with many programming languages.

Using software and programming to calculate statistics is more common for bigger sets of data, as finding it manually becomes difficult.
"""
"""
What Is a Confidence Interval?
A confidence interval, in statistics, refers to the probability that a population parameter will fall between a set of values for a certain proportion of times. Analysts often use confidence intervals that contain either 95% or 99% of expected observations. Thus, if a point estimate is generated from a statistical model of 10.00 with a 95% confidence interval of 9.50 - 10.50, it can be inferred that there is a 95% probability that the true value falls within that range.

Statisticians and other analysts use confidence intervals to understand the statistical significance of their estimations, inferences, or predictions. If a confidence interval contains the value of zero (or some other null hypothesis), then one cannot satisfactorily claim that a result from data generated by testing or experimentation is to be attributable to a specific cause rather than chance.
"""
# In[36]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=fifa19["Potential"])
plt.show()


# We can draw the vertical boxplot grouped by the categorical variable International Reputation
# as follows-

# In[37]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="International Reputation", y="Potential", data=fifa19)
plt.show()


# We can draw a boxplot with nested grouping by two categorical variables as follows-

# In[39]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="International Reputation", y="Potential", hue="Preferred Foot",data=fifa19, palette="Set3")
plt.show()


# Seaborn violinplot() function
# 
# • This function draws a combination of boxplot and kernel density estimate.
# 
# • A violin plot plays a similar role as a box and whisker plot.
# 
# • It shows the distribution of quantitative data across several levels of one (or more) categorical
# variables such that those distributions can be compared.
# 
# • Unlike a box plot, in which all of the plot components correspond to actual datapoints, the
# violin plot features a kernel density estimation of the underlying distribution.
# 
# • I will plot the violinplot of Potential variable as follows-

# In[40]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x=fifa19["Potential"])
plt.show()


# We can draw the vertical violinplot grouped by the categorical variable International Reputation
# as follows-

# In[41]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x="International Reputation", y="Potential", data=fifa19)
plt.show()


# We can draw a violinplot with nested grouping by two categorical variables as follows-

# In[44]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa19, palette="muted")
plt.show()


# We can draw split violins to compare the across the hue variable as follows-

# In[46]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x="International Reputation", y="Potential", hue="Preferred Foot",
data=fifa19, palette="muted", split=True)
plt.show()


# Seaborn pointplot() function
# 
# • This function show point estimates and confidence intervals using scatter plot glyphs.
# 
# • A point plot represents an estimate of central tendency for a numeric variable by the position
# of scatter plot points and provides some indication of the uncertainty around that estimate
# using error bars.

# In[48]:
"""
An Introduction to Point Estimation in Statistics
In Statistics, Estimation Theory and Hypothesis Testing play a major role in determining solutions to certain problems. Point estimation is one of the areas that help people involved in Statistical analysis come to conclusions regarding many different kinds of questions. Point estimation means using data to calculate the value or the point as it serves as a best guess of any given parameter that may be unknown. 


What is the Definition of Point Estimation?
Point estimators are defined as functions that can be used to find the approximate value of a particular point from a given population parameter. The sample data of a population is used to find a point estimate or a statistic that can act as the best estimate of an unknown parameter that is given for a population. 


What are the Properties of Point Estimators? 
It is desirable for a point estimate to be the following :

Consistent - We can say that the larger is the sample size, the more accurate is the estimate. 

Unbiased - The expectation of the observed values of various samples equals the corresponding population parameter. Let’s take, for example, We can say that sample mean is an unbiased estimator for the population mean.

Most Efficient That is also Known as Best Unbiased - of all the various consistent, unbiased estimates, the one possessing the smallest variance (a measure of the amount of dispersion away from the estimate). In simple words, we can say that the estimator varies least from sample to sample and this generally depends on the particular distribution of the population. For example, the mean is more efficient than the median (that is the middle value) for the normal distribution but not for more “skewed” ( also known as asymmetrical) distributions.


What are the Methods Used to Calculate Point Estimators?
The maximum likelihood method is a popularly used way to calculate point estimators. This method uses differential calculus to understand the probability function from a given number of sample parameters. 


Named after Thomas Bayes, the Bayesian method is another way using which the frequency function of a parameter can be understood. This is a more non-traditional approach. However, in this case, enough information on the distribution of the parameter is not always given but in case it is, then the estimation can be done fairly easily. 


What are the Formulae that Can be Used to Measure Point Estimators? 
Some common formulae include: 

Maximum Likelihood Estimation or MLE

Jeffrey Estimation

Wilson Estimation

Laplace Estimation


What are the Values Needed to Calculate Point Estimators?
The number of successes is shown by S.

The number of trials is shown by T.

The Z–score is shown by z. 


Once You Know All the Values Listed Above, You Can Start Calculating the Point Estimate According to the Following Given Equations:
Maximum Likelihood Estimation: MLE = S / T

Laplace Estimation: Laplace equals (S + 1) / (T + 2)

Jeffrey Estimation: Jeffrey equals (S + 0.5) / (T + 1)

Wilson Estimation: Wilson equals (S + z²/2) / (T + z²)


Once All Four Values have been Calculated, You Need to Choose the Most Accurate One.


This should be done According to the Following Rules Listed below:

If the value of  MLE ≤ 0.5, the Wilson Estimation is the most accurate.

If the value of MLE - 0.5 < MLE < 0.9, then the Maximum Likelihood Estimation is the most accurate.

If 0.9 < MLE, then the smaller of Jeffrey and Laplace Estimations is said to be the most accurate.


"""

f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential", data=fifa19)
plt.show()


# We can draw a set of vertical points with nested grouping by a two variables as follows-

# In[50]:


f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa19)
plt.show()


# We can separate the points for different hue levels along the categorical axis as follows-

# In[52]:


f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa19, dodge=True)
plt.show()


# We can use a different marker and line style for the hue levels as follows-

# In[53]:


f, ax = plt.subplots(figsize=(8, 6))
sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot",
data=fifa19, markers=["o", "x"], linestyles=["-", "--"])
plt.show()


#  Seaborn barplot() function
# • This function show point estimates and confidence intervals as rectangular bars.
# • A bar plot represents an estimate of central tendency for a numeric variable with the height
# of each rectangle and provides some indication of the uncertainty around that estimate using
# error bars.
# • Bar plots include 0 in the quantitative axis range, and they are a good choice when 0 is a
# meaningful value for the quantitative variable, and you want to make comparisons against it.
# • We can plot a barplot as follows-

# In[54]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa19)
plt.show()


# We can draw a set of vertical bars with nested grouping by a two variables as follows-

# In[56]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", hue="Preferred Foot",data=fifa19)
plt.show()


# We can use median as the estimate of central tendency as follows-

# In[57]:


from numpy import median
f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa19,estimator=median)
plt.show()


# We can show the standard error of the mean with the error bars as follows-

# In[58]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa19, ci=68)
plt.show()


# We can show standard deviation of observations instead of a confidence interval as follows-

# In[59]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa19, ci="sd")
plt.show()


# We can add “caps” to the error bars as follows-

# In[60]:


f, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa19, capsize=0.2)
plt.show()


# Visualizing statistical relationship with Seaborn relplot() function
# 1.0.26 Seaborn relplot() function
# • Seaborn relplot() function helps us to draw figure-level interface for drawing relational plots
# onto a FacetGrid.
# • This function provides access to several different axes-level functions that show the relationship between two variables with semantic mappings of subsets.
# • The kind parameter selects the underlying axes-level function to use-
# • scatterplot() (with kind=“scatter”; the default)
# • lineplot() (with kind=“line”)
# We can plot a scatterplot with variables Heigh and Weight with Seaborn relplot() function as
# follows

# In[62]:


g = sns.relplot(x="Overall", y="Potential", data=fifa19)


#  Seaborn scatterplot() function
# • This function draws a scatter plot with possibility of several semantic groups.
# • The relationship between x and y can be shown for different subsets of the data using the
# hue, size and style parameters.
# • These parameters control what visual semantics are used to identify the different subsets.

# In[63]:


f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="Height", y="Weight", data=fifa19)
plt.show()


# Seaborn lineplot() function
# • THis function draws a line plot with possibility of several semantic groupings.
# • The relationship between x and y can be shown for different subsets of the data using the
# hue, size and style parameters.
# • These parameters control what visual semantics are used to identify the different subsets.

# In[64]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.lineplot(x="Stamina", y="Strength", data=fifa19)
plt.show()


# Visualize linear relationship with Seaborn regplot() function
# 
# Seaborn regplot() function
seaborn.regplot() :
# This method is used to plot data and a linear regression model fit.
# There are a number of mutually exclusive options for estimating the regression model. 
# Syntax : seaborn.regplot( x,  y,  data=None, x_estimator=None, x_bins=None,  x_ci=’ci’, scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None, order=1, logistic=False, lowess=False, robust=False, logx=False, x_partial=None, y_partial=None, truncate=False, dropna=True, x_jitter=None, y_jitter=None, label=None, color=None, marker=’o’,    scatter_kws=None, line_kws=None, ax=None)
# • This function plots data and a linear regression model fit.
# 
# • We can plot a linear regression model between Overall and Potential variable with
#   regplot() function as follows-

# In[65]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="Overall", y="Potential", data=fifa19)
plt.show()


# We can use a different color and marker as follows-

# In[66]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="Overall", y="Potential", data=fifa19, color= "g",marker="+")
plt.show()


# We can plot with a discrete variable and add some jitter as follows-

# In[67]:


f, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x="International Reputation", y="Potential", data=fifa19, x_jitter=.01)
plt.show()


#  Seaborn lmplot() function
#     
# • This function plots data and regression model fits across a FacetGrid.
# 
# • This function combines regplot() and FacetGrid. 
# • It is intended as a convenient interface to fit regression models across conditional subsets of
# a dataset.
# 
# • We can plot a linear regression model between Overall and Potential variable with lmplot() function as follows-
# 

# In[68]:


g= sns.lmplot(x="Overall", y="Potential", data=fifa19)


# We can condition on a third variable and plot the levels in different colors as follows-

# In[69]:


g= sns.lmplot(x="Overall", y="Potential", hue="Preferred Foot", data=fifa19)


# We can use a different color palette as follows-

# In[70]:


g= sns.lmplot(x="Overall", y="Potential", hue="Preferred Foot", data=fifa19,palette="Set1")


# We can plot the levels of the third variable across different columns as follows-

# In[71]:


g= sns.lmplot(x="Overall", y="Potential", col="Preferred Foot", data=fifa19)


# Multi-plot grids
# Seaborn FacetGrid() function
# • The FacetGrid class is useful when you want to visualize the distribution of a variable or the
# relationship between multiple variables separately within subsets of your dataset.
# • A FacetGrid can be drawn with up to three dimensions - row, col and hue. The first two
# have obvious correspondence with the resulting array of axes - the hue variable is a third
# dimension along a depth axis, where different levels are plotted with different colors.
# • The class is used by initializing a FacetGrid object with a dataframe and the names of the
# variables that will form the row, column or hue dimensions of the grid.
# • These variables should be categorical or discrete, and then the data at each level of the
# variable will be used for a facet along that axis.
# We can initialize a 1x2 grid of facets using the fifa19 dataset.

# In[73]:


g = sns.FacetGrid(fifa19, col="Preferred Foot")


# We can draw a univariate plot of Potential variable on each facet as follows-

# In[74]:


g = sns.FacetGrid(fifa19, col="Preferred Foot") 
g = g.map(plt.hist, "Potential")


# In[75]:


g = sns.FacetGrid(fifa19, col="Preferred Foot") 
g = g.map(plt.hist, "Potential", bins=10, color="r")


# We can plot a bivariate function on each facet as follows-

# In[76]:


g = sns.FacetGrid(fifa19, col="Preferred Foot") 
g = (g.map(plt.scatter, "Height", "Weight", edgecolor="w").add_legend())


# The size of the figure is set by providing the height of each facet, along with the aspect ratio:

# In[77]:


g = sns.FacetGrid(fifa19, col="Preferred Foot", height=5, aspect=1) 
g = g.map(plt.hist, "Potential")


#  Seaborn Pairgrid() function
# • This function plots subplot grid for plotting pairwise relationships in a dataset.
# • This class maps each variable in a dataset onto a column and row in a grid of multiple axes.
# • Different axes-level plotting functions can be used to draw bivariate plots in the upper and
# 50
# lower triangles, and the the marginal distribution of each variable can be shown on the
# diagonal.
# • It can also represent an additional level of conditionalization with the hue parameter, which
# plots different subets of data in different colors.
# • This uses color to resolve elements on a third dimension, but only draws subsets on top of
# each other and will not tailor the hue parameter for the specific visualization the way that
# axes-level functions that accept hue will.

# In[79]:


fifa19_new = fifa19[['Age', 'Potential', 'Strength', 'Stamina', 'Preferred Foot']]


# In[80]:


g = sns.PairGrid(fifa19_new)
g = g.map(plt.scatter)


# We can show a univariate distribution on the diagonal as follows-

# In[81]:


g = sns.PairGrid(fifa19_new)
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)


# We can color the points using the categorical variable Preferred Foot as follows -

# In[83]:


g = sns.PairGrid(fifa19_new, hue="Preferred Foot") 
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()


# We can use a different style to show multiple histograms as follows-

# In[86]:


g = sns.PairGrid(fifa19_new, hue="Preferred Foot") 
g = g.map_diag(plt.hist, histtype="step", linewidth=3) 
g = g.map_offdiag(plt.scatter)
g = g.add_legend()


# We can plot a subset of variables as follows-

# In[87]:


g = sns.PairGrid(fifa19_new, vars=['Age', 'Stamina'])
g = g.map(plt.scatter)


# We can use different functions on the upper and lower triangles as follows-

# In[89]:


g = sns.PairGrid(fifa19_new)
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot, cmap="Blues_d") 
g = g.map_diag(sns.kdeplot, lw=3, legend=False)


# Seaborn Jointgrid() function
# • This function provides a grid for drawing a bivariate plot with marginal univariate plots.
# • It set up the grid of subplots.
# We can initialize the figure and add plots using default parameters as follows-

# In[90]:


g = sns.JointGrid(x="Overall", y="Potential", data=fifa19)
g = g.plot(sns.regplot, sns.distplot)


# We can draw the join and marginal plots separately, which allows finer-level control other parameters as follows -

# In[91]:


import matplotlib.pyplot as plt
g = sns.JointGrid(x="Overall", y="Potential", data=fifa19)
g = g.plot_joint(plt.scatter, color=".5", edgecolor="white") 
g = g.plot_marginals(sns.distplot, kde=False, color=".5")


# We can remove the space between the joint and marginal axes as follows -

# In[94]:


g = sns.JointGrid(x="Overall", y="Potential", data=fifa19, space=0) 
g = g.plot_joint(sns.kdeplot, cmap="Blues_d") 
g = g.plot_marginals(sns.kdeplot, shade=True)


# We can draw a smaller plot with relatively larger marginal axes as follows -

# In[95]:


g = sns.JointGrid(x="Overall", y="Potential", data=fifa19, height=5, ratio=2) 
g = g.plot_joint(sns.kdeplot, cmap="Reds_d") 
g = g.plot_marginals(sns.kdeplot, color="r", shade=True)


# Controlling the size and shape of the plot
# • The default plots made by regplot() and lmplot() look the same but on axes that have a
# different size and shape.
# • This is because regplot() is an “axes-level” function draws onto a specific axes.
# • This means that you can make multi-panel figures yourself and control exactly where the
# regression plot goes.
# • If no axes object is explicitly provided, it simply uses the “currently active” axes, which is
# why the default plot has the same size and shape as most other matplotlib functions.
# • To control the size, we need to create a figure object ourself as follows-

# In[96]:


f, ax = plt.subplots(figsize=(8, 6))
ax = sns.regplot(x="Overall", y="Potential", data=fifa19);


# In contrast, the size and shape of the lmplot() figure is controlled through the FacetGrid interface
# using the size and aspect parameters, which apply to each facet in the plot, not to the overall figure
# itself.

# In[97]:


sns.lmplot(x="Overall", y="Potential", col="Preferred Foot", data=fifa19,col_wrap=2, height=5, aspect=1)


# Seaborn figure styles
# • There are five preset seaborn themes: darkgrid, whitegrid, dark, white and ticks. • They are each suited to different applications and personal preferences.
# • The default theme is darkgrid.
# • The grid helps the plot serve as a lookup table for quantitative information, and the white-on
# grey helps to keep the grid from competing with lines that represent data.
# • The whitegrid theme is similar, but it is better suited to plots with heavy data elements:
# I will define a simple function to plot some offset sine waves, which will help us see the different
# stylistic parameters as follows -

# In[98]:


def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)


# This is what the plot looks like with matplotlib default parameters.

# In[99]:


sinplot()


# To switch to seaborn defaults, we need to call the set() function as follows -

# In[100]:


sns.set()
sinplot()


# • We can set different styles as follows -

# In[101]:


sns.set_style("whitegrid")
sinplot()
"""
seaborn.regplot() method is used to plot data and draw a linear regression model fit. There are several options for estimating the regression model, all of which are mutually exclusive.

As we might already know, Regrression Analysis is a technique used to evaluate the relationship between independent factors and dependent attributes. Hence, this model is used to create a regression plot.

The regplot() and lmplot() functions are relatively close, but the regplot() method is an axes level function while the other is not. Matplotlib axes containing the plot are returned as a result of this method.

Syntax
Following is the syntax of seaborn.regplot() method −

seaborn.regplot(*, x=None, y=None, data=None, x_estimator=None, x_bins=None, x_ci='ci', scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None, seed=None, order=1, logistic=False, lowess=False, robust=False, logx=False, x_partial=None, y_partial=None, truncate=True, dropna=True, x_jitter=None, y_jitter=None, label=None, color=None, marker='o', scatter_kws=None, line_kws=None, ax=None)
Parameters
Some of the parameters of the regplot() method are discussed below.

S.No	Parameter and Description
1	x,y
These parameters take names of variables as input that plot the long form data.

2	data
This is the dataframe that is used to plot graphs.

3	x_estimator
This is a callable that accepts values and maps vectors to scalars. It is an optional parameter. Each distinct value of x is applied to this function, and the estimated value is plotted as a result. When x is a discrete variable, this is helpful. This estimate will be bootstrapped and a confidence interval will be drawn if x_ci is provided.

4	x_bins
This optional parameter accepts int or vector as input. The x variable is binned into discrete bins and then the central tendency and confidence interval are estimated.

5	{x,y}_jitter
This optional parameter accepts floating point values. Add uniform random noise of this size to either the x or y variables.

6	color
Used to specify a single color, and this color is applied to all plot elements.

7	marker
This is the marker that is used to plot the data points in the graph.

8	x_ci
Takes values from ciâ€, â€œsdâ€, int in [0, 100] or None. It is an optional parameter.

The size of the confidence interval used when plotting a central tendency for discrete values of x is determined by the value passed to this parameter.

9	logx
Takes boolean vaules and if True, plots the scatterplot and regression model in the input space while also estimating a linear regression of the type y log(x). For this to work, x must be positive.
"""

# In[102]:


sns.set_style("dark")
sinplot()
"""
The Seaborn.set_style() method sets the parameters that control the general style of the plots. This method works closely with the seaborn.axes_style() method as this also checks whether the grid is enabled by default and uses its style parameters control various properties like background color etc.

Setting these parameters to control the general style of the plot can be accomplished by the matplotlib rcParams system.

Syntax
Following is the syntax of the seaborn.set_style() method −

seaborn.set_style(style=None, rc=None)
Parameters
Following are the parameters of seaborn.set_style() method −

S.No	Parameter and Description
1	Style
Takes values, None, dict, or one of {darkgrid, whitegrid, dark, white, ticks} and determines a dictionary of parameters or the name of a preconfigured style.

2	Rc
Takes rcdict as value and is an optional parameter that performs Parameter mappings to override the values in the preset seaborn style dictionaries. This only updates parameter that are considered part of the style definition.

Now we will move onto understanding the method and using it in examples.
"""


# In[103]:


sns.set_style("white")
sinplot()


# In[104]:


sns.set_style("ticks")
sinplot()
"""
The Seaborn.set_style() method sets the parameters that control the general style of the plots. This method works closely with the seaborn.axes_style() method as this also checks whether the grid is enabled by default and uses its style parameters control various properties like background color etc.

Setting these parameters to control the general style of the plot can be accomplished by the matplotlib rcParams system.

Syntax
Following is the syntax of the seaborn.set_style() method −

seaborn.set_style(style=None, rc=None)
Parameters
Following are the parameters of seaborn.set_style() method −

S.No	Parameter and Description
1	Style
Takes values, None, dict, or one of {darkgrid, whitegrid, dark, white, ticks} and determines a dictionary of parameters or the name of a preconfigured style.

2	Rc
Takes rcdict as value and is an optional parameter that performs Parameter mappings to override the values in the preset seaborn style dictionaries. This only updates parameter that are considered part of the style definition.

Now we will move onto understanding the method and using it in examples.
"""

# With that, we come to the end of this tutorial.
# I hope you find it useful.
# Please upvote it if it helps to learn Seaborn.

# In[ ]:




