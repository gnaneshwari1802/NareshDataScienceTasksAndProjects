import nltk
import re
import os
#nltk.download('all')
nltk.download('stopwords')
#from gensim.models import Word2Vec

from nltk.corpus import stopwords
stop_words = stopwords.words('Telugu')
"""stop_words.append("చేత")
stop_words.append("వలన")
stop_words.append("గూర్చి")

stop_words.append("కొరకు")
"""
print(stop_words)
