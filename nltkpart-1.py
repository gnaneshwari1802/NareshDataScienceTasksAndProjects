# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:49:04 2023

@author: M GNANESHWARI
"""

import nltk
nltk.download('all')
nltk.download('averaged_perceptron_tagger')

from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import blankline_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import wordpunct_tokenize
from nltk.util import bigrams, trigrams, ngrams
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.util import bigrams, trigrams, ngrams
#nltk.download('all')  # Download all NLTK data, including WordNet
#nltk.download('wordnet')
"""
# Check the path to WordNet data
wordnet_path = nltk.data.find('/usr/share/nltk_data/corpora/wordnet')
print("WordNet Data Path:", wordnet_path)
nltk.data.path.append('/usr/share/nltk_data/corpora/wordnet')
"""

# You don't need to set the WordNet data path manually.
# NLTK will find it automatically after you download it.

Se = '''In 2022, the tech industry saw remarkable growth. Companies like Apple, with a market cap exceeding $2.1 trillion,
unveiled new products, while startups secured billions in funding. The pandemic accelerated digital transformation, boosting
e-commerce sales by 27%. AI and IoT continued to shape our lives. Amid this, cyberattacks surged by 64%, highlighting the need
for robust cybersecurity. As we move forward, the fusion of technology and daily life is evident, promising both innovation and
challenges for the years ahead.'''
print(Se)

print(type(Se))
print("\n")

word_tokenize(Se)
print(len(word_tokenize(Se)))

print("\n")
print(sent_tokenize(Se))
print("\n")
print(blankline_tokenize(Se))
print("\n")
print(WhitespaceTokenizer().tokenize(Se))

print(wordpunct_tokenize(Se))
quote = 'The best and most beautiful things in the world cannot be seen or even touched, theymust be felt with the heart.'
quote_token = word_tokenize(quote)
print(quote_token)

print(list(nltk.bigrams(quote_token)))
print(list(nltk.trigrams(quote_token)))
print(list(nltk.ngrams(quote_token, 5)))


pst = PorterStemmer()
print(pst.stem('Affection'))

word_to_stem = ['giving', 'given', 'thanking', 'maximum', 'loving']
for words in word_to_stem:
    print(words + ' : ' + pst.stem(words))


lst = LancasterStemmer()
print(lst.stem('love'))
for words in word_to_stem:
    print(words + ' : ' + lst.stem(words))
print("\n")

sbst = SnowballStemmer('english')
print(sbst.stem('loving'))

for words in word_to_stem:
    print(words + ' : ' + sbst.stem(words))
print("\n")
word_lem = WordNetLemmatizer()
print(word_lem.lemmatize('loving'))
for words in word_to_stem:
    print(words + ' : ' + word_lem.lemmatize(words))
print("\n")
stopwords.words('english')
print(len(stopwords.words('english')))

stopwords.words('german')
print(len(stopwords.words('german')))
stopwords.words('chinese')

print(len(stopwords.words('chinese')))
