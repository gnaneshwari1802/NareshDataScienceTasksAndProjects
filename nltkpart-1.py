# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:49:04 2023

@author: M GNANESHWARI
"""

import os
import nltk
In [2]: Se = '''In 2022, the tech industry saw remarkable growth. Companies like Apple, with a market cap exceeding $2.1 trillion,
unveiled new products, while startups secured billions in funding. The pandemic accelerated digital transformation, boosting
e-commerce sales by 27%. AI and IoT continued to shape our lives. Amid this, cyberattacks surged by 64%, highlighting the need
for robust cybersecurity. As we move forward, the fusion of technology and daily life is evident, promising both innovation and
challenges for the years ahead.'''
Se
Out[2]: In [3]:
type(Se)
Out[3]: In [4]:
from nltk.tokenize import word_tokenize
word_tokenize(Se)