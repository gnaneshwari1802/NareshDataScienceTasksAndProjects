# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 18:35:20 2023

@author: M GNANESHWARI
"""

import nltk
nltk.download('punkt')


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
                
# Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)
print(sentences)
print('-'*20)
# Tokenizing words
words = nltk.word_tokenize(paragraph)
print(words)
output:
['AI, machine learning and deep learning are common terms in enterprise \n                IT and sometimes used interchangeably, especially by companies in their marketing materials.', 'But there are distinctions.', 'The term AI, coined in the 1950s, refers to the simulation of human \n                intelligence by machines.', 'It covers an ever-changing set of capabilities as new technologies \n                are developed.', 'Technologies that come under the umbrella of AI include machine learning and \n                deep learning.', 'Machine learning enables software applications to become more accurate at \n                predicting outcomes without being explicitly programmed to do so.', 'Machine learning algorithms \n                use historical data as input to predict new output values.', 'This approach became vastly more \n                effective with the rise of large data sets to train on.', 'Deep learning, a subset of machine \n                learning, is based on our understanding of how the brain is structured.', "Deep learning's \n                use of artificial neural networks structure is the underpinning of recent advances in AI, \n                including self-driving cars and ChatGPT."]
--------------------
['AI', ',', 'machine', 'learning', 'and', 'deep', 'learning', 'are', 'common', 'terms', 'in', 'enterprise', 'IT', 'and', 'sometimes', 'used', 'interchangeably', ',', 'especially', 'by', 'companies', 'in', 'their', 'marketing', 'materials', '.', 'But', 'there', 'are', 'distinctions', '.', 'The', 'term', 'AI', ',', 'coined', 'in', 'the', '1950s', ',', 'refers', 'to', 'the', 'simulation', 'of', 'human', 'intelligence', 'by', 'machines', '.', 'It', 'covers', 'an', 'ever-changing', 'set', 'of', 'capabilities', 'as', 'new', 'technologies', 'are', 'developed', '.', 'Technologies', 'that', 'come', 'under', 'the', 'umbrella', 'of', 'AI', 'include', 'machine', 'learning', 'and', 'deep', 'learning', '.', 'Machine', 'learning', 'enables', 'software', 'applications', 'to', 'become', 'more', 'accurate', 'at', 'predicting', 'outcomes', 'without', 'being', 'explicitly', 'programmed', 'to', 'do', 'so', '.', 'Machine', 'learning', 'algorithms', 'use', 'historical', 'data', 'as', 'input', 'to', 'predict', 'new', 'output', 'values', '.', 'This', 'approach', 'became', 'vastly', 'more', 'effective', 'with', 'the', 'rise', 'of', 'large', 'data', 'sets', 'to', 'train', 'on', '.', 'Deep', 'learning', ',', 'a', 'subset', 'of', 'machine', 'learning', ',', 'is', 'based', 'on', 'our', 'understanding', 'of', 'how', 'the', 'brain', 'is', 'structured', '.', 'Deep', 'learning', "'s", 'use', 'of', 'artificial', 'neural', 'networks', 'structure', 'is', 'the', 'underpinning', 'of', 'recent', 'advances', 'in', 'AI', ',', 'including', 'self-driving', 'cars', 'and', 'ChatGPT', '.']
[nltk_data] Downloading package punkt to /root/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
