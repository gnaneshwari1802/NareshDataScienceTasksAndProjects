# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:57:49 2023

@author: M GNANESHWARI
"""
import nltk
from nltk.data import find

# Download the NLTK data index
nltk.download('all', download_dir=find('C:\Python311\Lib\site-packages\nltk'))
nltk.data.path.append('C:\\Users\\M GNANESHWARI/nltk_data')
import nltk
from nltk.stem import WordNetLemmatizer
# wordnetlematizer library is the responsible for doing the lemmatization function
from nltk.corpus import stopwords
nltk.download()
paragraph = '''AI, machine learning and deep learning are common terms in enterprise 
                IT and sometimes used interchangeably, especially by companies in their marketing materials. 
                But there are distinctions. The term AI, coined in the 1950s, refers to the simulation of human 
                intelligence by machines. It covers an ever-changing set of capabilities as new technologies 
                are developed. Technologies that come under the umbrella of AI include machine learning and 
                deep learning. Machine learning enables software applications to become more accurate at 
                predicting outcomes without being explicitly programmed to do so. Machine learning algorithms 
                use historical data as input to predict new output values. This approach became vastly more 
                effective with the rise of large data sets to train on. Deep learning, a subset of machine 
                learning, is based on our understanding of how the brain is structured. Deep learning's 
                use of artificial neural networks structure is the underpinning of recent advances in AI, 
                including self-driving cars and ChatGPT.'''  
               
sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

# Lemmatization
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)      
    
    
