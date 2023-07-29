#!/usr/bin/env python
# coding: utf-8

# 0.0.1 IRIS DATASET VISUALIZATION(SEABORN,MATPLOTLIB)
# I have created this Kernel for beginners who want to learn how to plot graphs with
# seaborn.This kernel is still a work in progress.I will be updating it further when I
# find some time.If you find my work useful please fo vote by clicking at the top of the
# page.Thanks for viewing.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries␣ ↪installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/↪docker-python
# For example, here's several helpful packages to load in
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Importing pandas and Seaborn module

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore') 
#this will ignore the warnings.it wont␣ ↪display warnings in notebook


# Importing Iris data set

# In[4]:


iris=pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\Iris.csv')


# In[5]:


iris


# Displaying data

# In[5]:


iris.head()


# In[6]:


iris.drop('Id',axis=1,inplace=True)


# In[7]:


iris.head()


# In[8]:


iris.info()


# In[9]:


iris['Species'].value_counts()


# In[10]:


sns.countplot('Species',data=iris)
plt.show()


# We can see that there are 50 samples each of all the Iris Species in the data set.
# 4. Joint plot: Jointplot is seaborn library specific and can be used to quickly visualize and
# analyze the relationship between two variables and describe their individual distributions on the
# same plot.

# In[11]:


iris.head()


# In[12]:


fig=sns.jointplot(x='SepalLengthCm',y='SepalWidthCm',data=iris)


# In[13]:


sns.jointplot("SepalLengthCm", "SepalWidthCm", data=iris, kind="reg")


# In[14]:


fig=sns.jointplot(x='SepalLengthCm',y='SepalWidthCm',kind='hex',data=iris)


# 5. FacetGrid Plot

# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.FacetGrid(iris,hue='Species',size=5).map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend()


# 6. Boxplot or Whisker plot Box plot was was first introduced in year 1969 by Mathematician
# John Tukey.Box plot give a statical summary of the features being plotted.Top line represent the
# max value,top edge of box is third Quartile, middle edge represents the median,bottom edge represents the first quartile value.The bottom most line respresent the minimum value of the feature.The
# height of the box is called as Interquartile range.The black dots on the plot represent the outlier
# values in the data.

# In[16]:


iris.head()


# In[17]:


fig=plt.gcf()
fig.set_size_inches(10,7)


# In[19]:


fig=sns.boxplot(x='Species',y='PetalLengthCm',data=iris,order=['Iris-virginica','Iris-versicolor','Iris-setosa'],linewidth=2.5,orient='v',dodge=False)


# In[20]:


#iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))
iris.boxplot(by="Species", figsize=(12, 6))


# 7.Stripplot

# In[26]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.stripplot(x='Species', y='SepalLengthCm', data=iris,jitter=True,edgecolor='gray' ,size=8,palette='winter', orient='v') 


# 8.CombiningBoxandStripPlots

# In[31]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.boxplot(x='Species', y='SepalLengthCm', data=iris)
fig=sns.boxplot(x='Species', y='SepalLengthCm', data=iris)
fig=sns.stripplot(x='Species', y='SepalLengthCm', data=iris,jitter=True,edgecolor='gray')


# In[34]:


ax=sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax=sns.stripplot(x="Species", y="PetalLengthCm", data=iris,jitter=True,edgecolor="gray")


# In[36]:


ax=sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax=sns.stripplot(x="Species", y="PetalLengthCm", data=iris,jitter=True,edgecolor="gray")
plt.show()


# 9.ViolinPlot Itisusedtovisualizethedistributionofdataanditsprobabilitydistribution.This chartisacombinationofaBoxPlotandaDensityPlotthatisrotatedandplacedoneachside,toshowthedistributionshapeofthedata.Thethickblackbarinthecentrerepresentsthe interquartilerange,thethinblacklineextendedfromitrepresentsthe95%confdenceintervals,andthewhitedotisthemedian.BoxPlotsarelimitedintheirdisplayofthedata,astheirvisual simplicitytendstohidesignifcantdetailsabouthowvaluesinthedataaredistributed

# In[37]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.violinplot(x='Species', y='SepalLengthCm', data=iris)


# In[39]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species', y='PetalLengthCm', data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species', y='PetalWidthCm' ,data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species', y='SepalLengthCm', data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species', y='SepalWidthCm' ,data=iris)


# 10.PairPlot: A“pairsplot”isalsoknownasascatterplot,inwhichonevariableinthesame datarowismatchedwithanothervariable’svalue,likethis:Pairsplotsarejustelaborationson this,showingallvariablespairedwithalltheothervariables.

# In[40]:


sns.pairplot(data=iris,kind='scatter') 


# In[41]:


sns.pairplot(iris,hue='Species') ;


# 11.Heatmap Heatmapisusedtofndoutthecorrelationbetweendiferentfeaturesinthe dataset.Highpositiveornegativevalueshowsthatthefeatureshavehighcorrelation.Thishelpsus toselecttheparmetersformachinelearning.

# In[44]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.heatmap(iris.corr(),annot=True,cmap='cubehelix' ,linewidths=1,linecolor='k', square=True,mask=False,vmin=-1,vmax=1,cbar_kws={"orientation":  "vertical"},cbar=True)


# 12.Distributionplot: Thedistributionplotissuitableforcomparingrangeanddistributionfor groupsofnumericaldata.Dataisplottedasvaluepointsalonganaxis.Youcanchoosetodisplay onlythevaluepointstoseethedistributionofvalues,aboundingboxtoseetherangeofvalues,oracombinationofbothasshownhere.Thedistributionplotisnotrelevantfordetailedanalysis ofthedataasitdealswithasummaryofthedatadistribution.

# In[46]:


iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)


# 13.Swarmplot Itlooksabitlikeafriendlyswarmofbeesbuzzingabouttheirhive.More importantly,eachdatapointisclearlyvisibleandnodataareobscuredbyoverplotting.Abeeswarm plotimprovesupontherandomjitteringapproachtomovedatapointstheminimumdistanceaway fromoneanothertoavoidoverlays.Theresultisaplotwhereyoucanseeeachdistinctdatapoint,likeshowninbelowplot

# In[47]:


sns.set(style="darkgrid")
fig=plt.gcf()
fig.set_size_inches(10,7)
fig = sns.swarmplot(x="Species", y="PetalLengthCm" ,data=iris)


# In[50]:


sns.set(style="whitegrid")
fig=plt.gcf()
fig.set_size_inches(10,7)
ax = sns.violinplot(x="Species", y="PetalLengthCm" ,data=iris,inner=None)
ax = sns.swarmplot(x="Species", y="PetalLengthCm", data=iris,color="white", edgecolor="black")


# 17.LMPLot

# In[51]:


fig=sns.lmplot(x="PetalLengthCm" ,y="PetalWidthCm" ,data=iris)


# 18.FacetGrid

# In[52]:


sns.FacetGrid(iris,hue="Species", size=6)


# In[54]:


sns.FacetGrid(iris,hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm").add_legend()
plt.ioff()


# **22.FactorPlot**

# In[7]:


#f,ax=plt.subplots(1,2,figsize=(18,8))
sns.factorplot('Species' ,'SepalLengthCm', data=iris)
plt.ioff()
plt.show()
#sns.factorplot('Species','SepalLengthCm',data=iris,ax=ax[0][0])#sns.factorplot('Species','SepalWidthCm',data=iris,ax=ax[0][1])
#sns.factorplot('Species','PetalLengthCm',data=iris,ax=ax[1][0])#sns.factorplot('Species','PetalWidthCm',data=iris,ax=ax[1][1])


# **23.BoxenPlot**

# In[57]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.boxenplot(x='Species', y='SepalLengthCm' ,data=iris)


# 28.KDEPlot

# In[13]:


#Createakdeplotofsepal_lengthversussepalwidthforsetosaspeciesofflower.
sub=iris[iris['Species']=='Iris-setosa']
sns.kdeplot(data=sub[['SepalLengthCm', 'SepalWidthCm']],cmap="plasma",shade=True,shade_lowest=False)
plt.title('Iris-setosa') 
plt.xlabel('SepalLengthCm')
plt.ylabel('epalWidthCm') 


# 30.Dashboard

# In[15]:


sns.set_style( 'darkgrid' )
f,axes=plt.subplots(2,2,figsize=(15,15))
k1=sns.boxplot(x="Species", y="PetalLengthCm", data=iris,ax=axes[0,0])
k2=sns.violinplot(x='Species', y='PetalLengthCm' ,data=iris,ax=axes[0,1])
k3=sns.stripplot(x='Species', y='SepalLengthCm', data=iris,jitter=True,edgecolor='gray' ,size=8,palette='winter', orient='v', ax=axes[1,0])
#axes[1,1].hist(iris.hist,bin=10)
axes[1,1]. hist(iris.PetalLengthCm,bins=100)
#k2.set(xlim=(-1,0.8))
plt.show()


# InthedashboardwehaveshownhowtocreatemultipleplotstofoamadashboardusingPython.In thisplotwehavedemonstratedhowtoplotSeabornandMatplotlibplotsonthesameDashboard.31.StackedHistogram

# In[16]:


iris['Species'] = iris['Species']. astype( 'category') #iris.head()


# In[23]:


list1=list()
mylabels=list()
for gen in iris.Species.cat.categories:
    list1.append(iris[iris.Species==gen].SepalLengthCm)
mylabels.append(gen)
h=plt.hist(list1,bins=30,stacked=True,rwidth=1,label=mylabels)
plt.legend()
plt.show()


# In[26]:


#iris['SepalLengthCm']=iris['SepalLengthCm'].astype('category')#iris.head()
#iris.plot.area(y='SepalLengthCm',alpha=0.4,figsize=(12,6));
iris.plot.area(y=['SepalLengthCm' ,'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],alpha=0.4,figsize=(12, 6));


# WithStackedHistogramwecanseethedistributionofSepalLengthofDiferentSpeciesto-gether.ThisshowsustherangeofSepanLengthforthethreediferentSpeciesofIrisFlower.
# 32.AreaPlot: AreaPlotgivesusavisualrepresentationofVariousdimensionsofIrisfowerand theirrangeindataset.

# 33.Distplot: Ithelpsustolookatthedistributionofasinglevariable.Kdeshowsthedensityof thedistribution

# In[72]:


sns.distplot(iris['SepalLengthCm'],kde=True,bins=20);


# #THISISALLABOUTEDACOMPLETE

# In[ ]:




