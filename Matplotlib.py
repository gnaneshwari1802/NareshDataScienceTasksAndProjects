#!/usr/bin/env python
# coding: utf-8

# Import Matplotlib

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[76]:


get_ipython().run_line_magic('matplotlib', 'notebook')
"""
Magic commands, also known as magics, are one of the improved distinctions that IPython has compared to the classic Python shell. Magics are introduced to ease the efforts of commonly occurring tasks in data analysis using Python. They majorly determine the behavior of the IPython session.

matplotlib inline is a magic command that acts as a convenience function instead of clumsy python code to perform some configuration for the matplotlib in the current IPython session.
What is matplotlib inline in Python?
In IPython, there are various built-in functions known as magic functions. The magic functions are also called magic commands as they have a specific syntax used to call these magic functions.

The magic commands are of two kinds

line-oriented magic commands
cell-oriented magic commands
Line-Oriented Magic Commands
The line-oriented magic commands are also known as line magics. The syntax of line magics starts with a single percentage % followed by a command name followed by arguments. The command after % sign analogy can be seen with the command line interface of the operating system. The line magics do not contain any parenthesis or quotes.

Examples of line magics are: - %autowait, - %config, - %alias, - %conda, - %debug, - %load, - %notebook, - %macro, - %matplotlib, - so on
Some line magics have returns values that can be assigned to some variable and further used.
"""

# Displaying Plots in Matplotlib

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')

x1=np.linspace(0,10,100)
#create a plot figure
plt.plot(x1,np.sin(x1),'-')
plt.plot(x1,np.cos(x1),'--')
plt.figure()


# Pyplot API

# In[75]:


plt.gcf ( ) # get current figure
plt.gca ( ) # get current axes


# In[8]:


plt.figure()


# In[9]:


#create the first of two panels and set current axis
plt.subplot(2,1,1) #(rows,columns,panel number)
plt.plot(x1,np.sin(x1))
#create the second of two panels and set current axis
plt.subplot(2,1,2) #(rows,columns,panel number)
plt.plot(x1,np.cos(x1))


# In[10]:


#get current figure information
print(plt.gcf())


# In[11]:


#get current axis information
print(plt.gca())


# Visualization with Pyplot

# In[12]:


plt.plot([1,2,3,4])


# In[14]:


plt.plot([1,2,3,4])
plt.ylabel('Numbers')
plt.show()


# In[16]:


plt.plot([1,2,3,4],[1,4,9,16])
plt.show()


# In[17]:


x=np.linspace(0,2,100)
plt.plot(x,x,label='linear')
plt.plot(x,x**2,label='quadratic')
plt.plot(x,x**3,label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()
plt.show()


# In[18]:


plt.plot([1,2,3,4],[1,4,9,16],'ro')
plt.axis([0,6,0,20])
plt.show()


# In[21]:


#evenly sampled time at 200ms intervals
t=np.arange(0.,5.,0.2)
# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()


# In[22]:


# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)
# Call plot() method on the appropriate object
ax[0].plot(x1, np.sin(x1), 'b-')
ax[1].plot(x1, np.cos(x1), 'b-');


# In[23]:


fig = plt.figure()
x2 = np.linspace(0, 5, 10)
y2 = x2 ** 2
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(x2, y2, 'r')
axes.set_xlabel('x2')
axes.set_ylabel('y2')
axes.set_title('title');


# In[24]:


fig = plt.figure()
ax = plt.axes()


# In[25]:


ax1 = fig.add_subplot(2, 2, 1)


# In[74]:


fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)


# First plot with Matplotlib

# In[27]:


plt.plot([1, 3, 2, 4], 'b-')
plt.show( )


# In[28]:


plt.plot([1, 3, 2, 4], 'b-')


# Specify both Lists

# In[29]:


x3 = range(6)
plt.plot(x3, [xi**2 for xi in x3])
plt.show()


# In[30]:


x3 = np.arange(0.0, 6.0, 0.01)
plt.plot(x3, [xi**2 for xi in x3], 'b-')
plt.show()


# Multiline Plots

# In[31]:


x4 = range(1, 5)
plt.plot(x4, [xi*1.5 for xi in x4])
plt.plot(x4, [xi*3 for xi in x4])
plt.plot(x4, [xi/3.0 for xi in x4])
plt.show()


# Saving the Plot

# In[34]:


from IPython.display import Image
Image('fig1.png')


# In[35]:


# Saving the figure
fig.savefig('plot1.png')


# In[36]:


# Explore the contents of figure
from IPython.display import Image
Image('plot1.png')


# In[37]:


# Explore supported file formats
fig.canvas.get_supported_filetypes()


# In[38]:


# Create figure and axes first
fig = plt.figure()
ax = plt.axes()
# Declare a variable x5


# Line Plot

# In[40]:


# Create figure and axes first
fig = plt.figure()
ax = plt.axes()
# Declare a variable x5
x5 = np.linspace(0, 10, 1000)
# Plot the sinusoid function
ax.plot(x5, np.sin(x5), 'b-');


# Scatter Plot

# In[41]:


x7 = np.linspace(0, 10, 30)
y7 = np.sin(x7)
plt.plot(x7, y7, 'o', color = 'black');


# Histogram

# In[42]:


data1 = np.random.randn(1000)
plt.hist(data1)


# Bar Chart

# In[43]:


data2 = [5. , 25. , 50. , 20.]
plt.bar(range(len(data2)), data2)
plt.show()


# Horizontal Bar Chart

# In[44]:


data2 = [5. , 25. , 50. , 20.]
plt.barh(range(len(data2)), data2)
plt.show()


# Error Bar Chart

# In[45]:


x9 = np.arange(0, 4, 0.2)
y9 = np.exp(-x9)
e1 = 0.1 * np.abs(np.random.randn(len(y9)))
plt.errorbar(x9, y9, yerr = e1, fmt = '.-')
plt.show();


# Stacked Bar Chart

# In[46]:


A = [15., 30., 45., 22.]
B = [15., 25., 50., 20.]
z2 = range(4)
plt.bar(z2, A, color = 'b')
plt.bar(z2, B, color = 'r', bottom = A)
plt.show()


# In[47]:


A = [15., 30., 45., 22.]
B = [15., 25., 50., 20.]
z2 = range(4)
plt.bar(z2, A, color = 'b')
plt.bar(z2, B, color = 'r')
plt.show()


# Pie Chart

# In[48]:


plt.figure(figsize=(7,7))
x10 = [35, 25, 20, 20]
labels = ['Computer', 'Electronics', 'Mechanical', 'Chemical']
plt.pie(x10, labels=labels);
plt.show()


# Boxplot

# In[49]:


data3 = np.random.randn(100)
plt.boxplot(data3)
plt.show();


# Area Chart

# In[50]:


# Create some data
x12 = range(1, 6)
y12 = [1, 4, 6, 8, 4]
# Area plot
plt.fill_between(x12, y12)
plt.show()


# In[51]:


plt.stackplot(x12, y12)


# Contour Plot

# In[52]:


# Create a matrix
matrix1 = np.random.rand(10, 20)
cp = plt.contour(matrix1)
plt.show()


# Styles with Matplotlib Plots

# In[58]:


# View list of all available styles
print(plt.style.available)


# In[59]:


# Set styles for plots
plt.style.use('seaborn-bright')


# Adding a grid

# In[60]:


x15 = np.arange(1, 5)
plt.plot(x15, x15*1.5, x15, x15*3.0, x15, x15/3.0)
plt.grid(True)
plt.show()


# Handling axes

# In[61]:


x15 = np.arange(1, 5)
plt.plot(x15, x15*1.5, x15, x15*3.0, x15, x15/3.0)
plt.axis() # shows the current axis limits values
plt.axis([0, 5, -1, 13])
plt.show()


# In[62]:


x15 = np.arange(1, 5)
plt.plot(x15, x15*1.5, x15, x15*3.0, x15, x15/3.0)
plt.xlim([1.0, 4.0])
plt.ylim([0.0, 12.0])


# Handling X and Y ticks

# In[63]:


u = [5, 4, 9, 7, 8, 9, 6, 5, 7, 8]
plt.plot(u)
plt.xticks([2, 4, 6, 8, 10])
plt.yticks([2, 4, 6, 8, 10])
plt.show()


# Adding labels

# In[64]:


plt.plot([1, 3, 2, 4])
plt.xlabel('This is the X axis')
plt.ylabel('This is the Y axis')
plt.show()


# Adding a title

# In[66]:


plt.plot([1, 3, 2, 4])
plt.title('First Plot')
plt.show()


# Adding a legend

# In[67]:


x15 = np.arange(1, 5)
fig, ax = plt.subplots()
ax.plot(x15, x15*1.5)
ax.plot(x15, x15*3.0)
ax.plot(x15, x15/3.0)
ax.legend(['Normal','Fast','Slow'])


# In[68]:


x15 = np.arange(1, 5)
fig, ax = plt.subplots()
ax.plot(x15, x15*1.5, label='Normal')
ax.plot(x15, x15*3.0, label='Fast')
ax.plot(x15, x15/3.0, label='Slow')
ax.legend();


# Control colours

# In[69]:


x16 = np.arange(1, 5)
plt.plot(x16, 'r')
plt.plot(x16+1, 'g')
plt.plot(x16+2, 'b')
plt.show()


# Control line styles

# In[71]:


x16 = np.arange(1, 5)
plt.plot(x16, '--', x16+1, '-.', x16+2, ':')
plt.show()


# In[ ]:




