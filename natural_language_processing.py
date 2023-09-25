# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"/kaggle/input/restaurant-reviews/Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)
print(dataset)
# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
"""
What is the NLTK corpus?
A corpus is a collection of papers written in the same language. It will be a collection of text files stored in a directory, frequently surrounded by other text file directories. In the nltk. data.
Stop words are words that are so common they are basically ignored by typical tokenizers. By default, NLTK (Natural Language Toolkit) includes a list of 40 stop words, including: “a”, “an”, “the”, “of”, “in”, etc. The stopwords in nltk are the most common words in data.
"""
"""
What is Porter Stemmer in NLTK?
Porter Stemmer. This is the Porter stemming algorithm. It follows the algorithm presented in. Porter, M. “An algorithm for suffix stripping.” Program 14.3 (1980): 130-137.
"""
"""
What is the Porter algorithm for stemming?
Porter's Stemmer algorithm

It is one of the most popular stemming methods proposed in 1980. It is based on the idea that the suffixes in the English language are made up of a combination of smaller and simpler suffixes. This stemmer is known for its speed and simplicity.
"""
"""
How do you use Porter Stemmer in NLTK?
Create an instance of the PorterStemmer class. Define a sample sentence to be stemmed. Tokenize the sentence into individual words using word_tokenize. Use reduce to apply the PorterStemmer to each word in the tokenized sentence, and join the stemmed words back into a string.
"""
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] 
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)    
print("\n")  
"""
What is Sklearn Feature_extraction text?
The sklearn. feature_extraction module can be used to extract features in a format supported by machine learning algorithms from datasets consisting of formats such as text and image.
What is TfidfVectorizer?
The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features. FastText and Word2Vec Word Embeddings Python Implementation.
"""
# Creating the Bag of Words model
"""
What is Sklearn Feature_extraction text?
The sklearn. feature_extraction module can be used to extract features in a format supported by machine learning algorithms from datasets consisting of formats such as text and image.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
print(X,y)
print("\n") 
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
print(X_train, X_test, y_train, y_test)
# Training the Naive Bayes model on the Training set
print("\n") 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
print("\n") 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n") 
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)
print("\n")   
bias = classifier.score(X_train,y_train)
print(bias)
print("\n") 
variance = classifier.score(X_test,y_test)
print(variance)
print("\n") 
#===============================================

'''
CASE STUDY --> model is underfitted  & we got less accuracy 

1> Implementation of tfidf vectorization , lets check bias, variance, ac, auc, roc 
2> Impletemation of all classification algorihtm (logistic, knn, randomforest, decission tree, svm, xgboost) with bow & tfidf 
4> You can also reduce or increase test sample 
5> Try to apply k-fold cv 
6> Apply all overfitting techniques 
7> Try with GBM & LGBM as well
8> then please add more recores to train the data more records 
9- ac ,bias, varian - need to equal scale ( no overfit & not underfitt)

'''
