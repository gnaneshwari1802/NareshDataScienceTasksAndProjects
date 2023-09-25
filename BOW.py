import nltk
"""nltk.download('all')"""
nltk.download('averaged_perceptron_tagger')
import re # re libray will use for regular expression 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
"""
What is WordNetLemmatizer () in Python?
NLTK (Natural Language Toolkit) is a Python library used for natural language processing. One of its modules is the WordNet Lemmatizer, which can be used to perform lemmatization on words. Lemmatization is the process of reducing a word to its base or dictionary form, known as the lemma.
"""

paragraph =  """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""
print(paragraph)
print("\n")           
# Cleaning the texts

"""
What is a Porter stemmer?
Porter stemmer. The Porter stemming algorithm is a process for removing suffixes from words in English. Removing suffixes. automatically is an operation which is especially useful in the field of information retrieval.
"""
ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
"""
What is the use of Sent_tokenize?
The sent_tokenize function in Python can tokenize inserted text into sentences. In Python, we can tokenize with the help of the Natural Language Toolkit ( NLTK ) library. The library needs to be imported in the code.
"""
print(sentences) 
print("\n") 
corpus = []

# Create the empty list name as corpus becuase after cleaned the data corpus will store this clean data

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
#   review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]   
    review = ' '.join(review)
    corpus.append(review)
print(corpus)    
print("\n") 
# Creating the Bag of Words model 

# Also we called as document matrix 
"""
Basics of CountVectorizer
Learn everything you need to know about CountVectorizer.
Pratyaksh Jain
Towards Data Science
Pratyaksh Jain


·
Follow

Published in
Towards Data Science

·
7 min read
·
May 24, 2021
330


4






Photo by Ali Shah Lakhani on Unsplash
Machines cannot understand characters and words. So when dealing with text data we need to represent it in numbers to be understood by the machine. Countvectorizer is a method to convert text to numerical data. To show you how it works let’s take an example:

text = [‘Hello my name is james, this is my python notebook’]
The text is transformed to a sparse matrix as shown below.


We have 8 unique words in the text and hence 8 different columns each representing a unique word in the matrix. The row represents the word count. Since the words ‘is’ and ‘my’ were repeated twice we have the count for those particular words as 2 and 1 for the rest.

Countvectorizer makes it easy for text data to be used directly in machine learning and deep learning models such as text classification.

Let’s take another example, but this time with more than 1 input:

text = [‘Hello my name is james' , ’this is my python notebook’]
I have 2 text inputs, what happens is that each input is preprocessed, tokenized, and represented as a sparse matrix. By default, Countvectorizer converts the text to lowercase and uses word-level tokenization.


Now that we have looked at a few examples lets actually code!

We’ll first start by importing the necessary libraries. We’ll use the pandas library to visualize the matrix and the sklearn.feature_extraction.text which is a sklearn library to perform vectorization.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
text = [‘Hello my name is james’,
‘james this is my python notebook’,
‘james trying to create a big dataset’,
‘james of words to try differnt’,
‘features of count vectorizer’]
coun_vect = CountVectorizer()
count_matrix = coun_vect.fit_transform(text)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
print(df)

Parameters
Lowercase
Convert all characters to lowercase before tokenizing. Default is set to true and takes boolean value.

text = [‘hello my name is james’,
‘Hello my name is James’]
coun_vect = CountVectorizer(lowercase=False)
count_matrix = coun_vect.fit_transform(text)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
print(df)

Now lets try without using ‘lowercase = False’

text = [‘hello my name is james’,
‘Hello my name is James’]
coun_vect = CountVectorizer()
count_matrix = coun_vect.fit_transform(text)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
print(df)

Stop_words
Stopwords are the words in any language which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence. There are 3 ways of dealing with stopwords:

Custom stop words list
text = [‘Hello my name is james’,
‘james this is my python notebook’,
‘james trying to create a big dataset’,
‘james of words to try differnt’,
‘features of count vectorizer’]
coun_vect = CountVectorizer(stop_words= [‘is’,’to’,’my’])
count_matrix = coun_vect.fit_transform(text)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
print(df)
Sparse matrix after removing the words is , to and my:


2. sklearn built in stop words list

text = [‘Hello my name is james’,
‘james this is my python notebook’,
‘james trying to create a big dataset’,
‘james of words to try differnt’,
‘features of count vectorizer’]
coun_vect = CountVectorizer(stop_words=’english’)
count_matrix = coun_vect.fit_transform(text)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
print(df)

3. Using max_df and min_df (covered later)

Max_df:

Max_df stands for maximum document frequency. Similar to min_df, we can ignore words which occur frequently. These words could be like the word ‘the’ that occur in every document and does not provide and valuable information to our text classification or any other machine learning model and can be safely ignored. Max_df looks at how many documents contain the word and if it exceeds the max_df threshold then it is eliminated from the sparse matrix. This parameter can again 2 types of values, percentage and absolute.

Using absolute values:

text = [‘Hello my name is james’,
‘james this is my python notebook’,
‘james trying to create a big dataset’,
‘james of words to try differnt’,
‘features of count vectorizer’]
coun_vect = CountVectorizer(max_df=1)
count_matrix = coun_vect.fit_transform(text)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
print(df)
The words ‘is’, ‘to’, ‘james’, ‘my’ and ‘of’ have been removed from the sparse matrix as they occur in more than 1 document.


Using percentage:

text = [‘Hello my name is james’,
‘james this is my python notebook’,
‘james trying to create a big dataset’,
‘james of words to try differnt’,
‘features of count vectorizer’]
coun_vect = CountVectorizer(max_df=0.75)
count_matrix = coun_vect.fit_transform(text)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
print(df)
As you can see the word ‘james’ appears in 4 out of 5 documents(85%) and hence crosses the threshold of 75% and removed from the sparse matrix


Min_df:

Min_df stands for minimum document frequency, as opposed to term frequency which counts the number of times the word has occurred in the entire dataset, document frequency counts the number of documents in the dataset (aka rows or entries) that have the particular word. When building the vocabulary Min_df ignores terms that have a document frequency strictly lower than the given threshold. For example in your dataset you may have names that appear in only 1 or 2 documents, now these could be ignored as they do not provide enough information on the entire dataset as a whole but only a couple of particular documents. min_df can take absolute values(1,2,3..) or a value representing a percentage of documents(0.50, ignore words appearing in 50% of documents)

Using absolute values:

text = [‘Hello my name is james’,
‘james this is my python notebook’,
‘james trying to create a big dataset’,
‘james of words to try differnt’,
‘features of count vectorizer’]
coun_vect = CountVectorizer(min_df=2)
count_matrix = coun_vect.fit_transform(text)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
print(df)

max_features
The CountVectorizer will select the words/features/terms which occur the most frequently. It takes absolute values so if you set the ‘max_features = 3’, it will select the 3 most common words in the data.

text = [‘This is the first document.’,’This document is the second document.’,’And this is the third one.’, ‘Is this the first document?’,]
coun_vect = CountVectorizer(max_features=3)
count_matrix = coun_vect.fit_transform(text)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
print(df)
df
Binary
By setting ‘binary = True’, the CountVectorizer no more takes into consideration the frequency of the term/word. If it occurs it’s set to 1 otherwise 0. By default, binary is set to False. This is usually used when the count of the term/word does not provide useful information to the machine learning model.

text = [‘This is the first document. Is this the first document?’ ]
coun_vect = CountVectorizer(binary=True)
count_matrix = coun_vect.fit_transform(text)
count_array = count_matrix.toarray()
df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
print(df)
Even though all the words occur twice in the above input our sparse matrix just represents it with 1


Now let’s see if we used the default value:


Vocabulary
They are the collection of words in the sparse matrix.

text = [‘hello my name is james’,
‘Hello my name is James’]
coun_vect = CountVectorizer()
count_matrix = coun_vect.fit_transform(text)
print(coun_vect.vocabulary_)
The numbers do not represent the count of the words but the position of the words in the matrix

If you just want the vocabulary without the position of the word in the sparse matrix, you can use the method ‘get_feature_names()’. If you notice this is the same method we use while creating our database and setting our columns.

text = [‘Hello my name is james’,
‘james this is my python notebook’,
‘james trying to create a big dataset’]
coun_vect = CountVectorizer()
count_matrix = coun_vect.fit_transform(text)
print( coun_vect.get_feature_names())

CountVectorizer is just one of the methods to deal with textual data. Td-idf is a better method to vectorize data. I’d recommend you check out the official document of sklearn for more information.


"""
"""
What is Sklearn Feature_extraction text?
The sklearn. feature_extraction module can be used to extract features in a format supported by machine learning algorithms from datasets consisting of formats such as text and image.
"""
from sklearn.feature_extraction.text import CountVectorizer
"""
Countvectorizer is a method to convert text to numerical data. To show you how it works let's take an example: text = ['Hello my name is james, this is my python notebook'] The text is transformed to a sparse matrix as shown below.
"""
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
print(X)  
print("\n") 
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
X_tf = tf.fit_transform(corpus).toarray()
print(X_tf) 
