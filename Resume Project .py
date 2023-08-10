#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/code/harunshimanto/pandas-with-data-science-ai

# In[ ]:


get_ipython().run_cell_magic('html', '', "<style>\n@import url('https://fonts.googleapis.com/css?family=Ewert|Roboto&effect=3d|ice|');\nbody {background-color: gainsboro;} \na {color: #37c9e1; font-family: 'Roboto';} \nh1 {color: #37c9e1; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;} \nh2, h3 {color: slategray; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;}\nh4 {color: #818286; font-family: 'Roboto';}\nspan {font-family:'Roboto'; color:black; text-shadow: 5px 5px 5px #aaa;}  \ndiv.output_area pre{font-family:'Roboto'; font-size:110%; color:lightblue;}      \n</style>")


# 📋 Introduction of MoveLens:
# This is a report on the movieLens dataset available here. MovieLens itself is a research site run by GroupLens Research group at the University of Minnesota. The first automated recommender system was developed there in 1993.
# 
# 📝 Dataset Description:
# The dataset is available in several snapshots. The ones that were used in this analysis were Latest Datasets - both full and small (for web scraping). They were last updated in October 2016.
# 
# 📖 Definitions of Pandas:¶
# Pandas is a Python library for data analysis. It offers a number of data exploration, cleaning and transformation operations that are critical in working with data in Python.
# 
# Pandas build upon numpy and scipy providing easy-to-use data structures and data manipulation functions with integrated indexing.
# 
# The main data structures pandas provides are Series and DataFrames.

# Getting Started
# To get started, we will need to; Please note that you will need to download the dataset.
# 
# Here are the links to the data source and location:
# 
# Data Source: Kaggle Data Science Home (filename: movelens-20m-dataset.zip)
# Location: https://www.kaggle.com/grouplens/movielens-20m-dataset

# In[2]:


import pandas as pd


# Read the Dataset
# In this notebook, we will be using three CSV files:
# 
# ratings.csv : userId,movieId,rating, timestamp
# 
# tags.csv : userId,movieId, tag, timestamp
# 
# movies.csv : movieId, title, genres

# In[7]:


movies = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\movie.csv')
print(type(movies))
movies.head(20)


# ▩ DataFrames

# In[8]:


tags = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\tag.csv')
tags.head()


# In[9]:


ratings = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\rating.csv')
ratings.head()


# In[10]:


del ratings['timestamp']
del tags['timestamp']


# In[11]:


movies


# In[12]:


tags


# In[13]:


ratings


# In[28]:


row_0 = tags.iloc[0]
print(type(row_0))
print(row_0)
print(row_0.index)
print(row_0['userId'])
print('rating' in row_0)
print(row_0.name)
row_0 = row_0.rename('firstRow')
row_0.name


# 📦 Data Structures:
# 🚦 Series

# In[15]:


row_0 = ratings.iloc[0]
type(row_0)


# In[16]:


row_0 = movies.iloc[0]
type(row_0)


# In[17]:


print(row_0)


# In[29]:


tags.head()


# In[30]:


tags.index


# In[31]:


tags.columns


# In[32]:


tags.iloc[ [0,11,500] ]


# 📈 📉 Descriptive Statistics
# Let's look how the ratings are distributed!

# In[33]:


ratings['rating'].describe()


# In[34]:


ratings.describe()


# In[35]:


ratings['rating'].mean()


# In[36]:


ratings.mean()


# In[37]:


ratings['rating'].min()


# In[38]:


ratings['rating'].max()


# In[39]:


ratings['rating'].std()


# In[40]:


ratings['rating'].mode()


# In[41]:


ratings.corr()


# In[42]:


filter1 = ratings['rating'] > 10
print(filter1)
filter1.any()


# In[43]:


filter2 = ratings['rating'] > 0
filter2.all()


# 🔧 Data Cleaning: Handling Missing Data

# In[44]:


movies.shape


# In[45]:


movies.isnull().any()


# In[46]:


movies.isnull().any().any()


# In[47]:


ratings.shape


# In[48]:


ratings.isnull().any().any()


# In[49]:


tags.shape


# In[50]:


tags.isnull().any().any()


# In[51]:


tags=tags.dropna()


# In[52]:


tags.isnull().any().any()


# In[53]:


tags.shape


# 📊 Data Visualization

# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')

ratings.hist(column='rating', figsize=(10,5))


# In[55]:


ratings.boxplot(column='rating', figsize=(10,5))


# 📤 Slicing Out Columns

# In[56]:


tags['tag'].head()


# In[57]:


movies[['title','genres']].head()


# In[58]:


ratings[-10:]


# In[59]:


tag_counts = tags['tag'].value_counts()
tag_counts[-10:]


# In[60]:


tag_counts[:10].plot(kind='bar', figsize=(10,5))


# 🎣 Filters for Selecting Rows

# In[61]:


is_highly_rated = ratings['rating'] >= 5.0
ratings[is_highly_rated][30:50]


# In[62]:


is_action= movies['genres'].str.contains('Action')
movies[is_action][5:15]


# In[63]:


movies[is_action].head(15)


# In[64]:


ratings_count = ratings[['movieId','rating']].groupby('rating').count()
ratings_count


# In[65]:


average_rating = ratings[['movieId','rating']].groupby('movieId').mean()
average_rating.head()


# In[66]:


movie_count = ratings[['movieId','rating']].groupby('movieId').count()
movie_count.head()


# In[67]:


movie_count = ratings[['movieId','rating']].groupby('movieId').count()
movie_count.tail()


# In[68]:


tags.head()


# In[69]:


movies.head()


# In[70]:


t = movies.merge(tags, on='movieId', how='inner')
t.head()


# In[71]:


avg_ratings= ratings.groupby('movieId', as_index=False).mean()
del avg_ratings['userId']
avg_ratings.head()


# In[72]:


box_office = movies.merge(avg_ratings, on='movieId', how='inner')
box_office.tail()


# In[73]:


is_highly_rated = box_office['rating'] >= 4.0
box_office[is_highly_rated][-5:]


# In[74]:


is_Adventure = box_office['genres'].str.contains('Adventure')
box_office[is_Adventure][:5]


# In[75]:


box_office[is_Adventure & is_highly_rated][-5:]


# In[76]:


movies.head()


# In[77]:


movie_genres = movies['genres'].str.split('|', expand=True)


# In[78]:


movie_genres[:10]


# In[79]:


movie_genres['isComedy'] = movies['genres'].str.contains('Comedy')


# In[80]:


movie_genres[:10]


# In[81]:


movies['year'] = movies['title'].str.extract('.*\((.*)\).*', expand=True)


# In[82]:


movies.tail()


# In[83]:


tags = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\tag.csv')


# In[84]:


tags.dtypes


# In[85]:


tags.head(5)


# In[86]:


tags['parsed_time'] = pd.to_datetime(tags['timestamp'], unit='s')


# In[87]:


tags['parsed_time'].dtype


# In[88]:


tags.head(2)


# In[89]:


greater_than_t = tags['parsed_time'] > '2015-02-01'

selected_rows = tags[greater_than_t]

tags.shape, selected_rows.shape


# In[90]:


tags.sort_values(by='parsed_time', ascending=True)[:10]


# 📇 Average Movie Ratings over Time
# Movie ratings related to the year of launch?

# In[91]:


average_rating = ratings[['movieId','rating']].groupby('movieId', as_index=False).mean()
average_rating.tail()


# In[92]:


joined = movies.merge(average_rating, on='movieId', how='inner')
joined.head()
joined.corr()


# In[94]:


t = tags[pd.to_numeric(tags['movieId'], errors='coerce').notna()]


# In[95]:


t


# In[96]:


tags


# In[97]:


tags['movieId'] = pd.to_numeric(tags['movieId'], errors='coerce').fillna(-1).astype(int)
tags


# In[ ]:




