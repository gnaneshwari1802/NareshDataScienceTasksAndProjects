# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:49:04 2023

@author: M GNANESHWARI
"""

import nltk
nltk.download('all')
nltk.download('averaged_perceptron_tagger')
"""
WordNet is a lexical database of semantic relations between words that links words into semantic relations including synonyms, hyponyms, and meronyms. The synonyms are grouped into synsets with short definitions and usage examples. It can thus be seen as a combination and extension of a dictionary and thesaurus.
"""
"""
Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item. Lemmatization is similar to stemming but it brings context to the words. So it links words with similar meanings to one word. 
Text preprocessing includes both Stemming as well as Lemmatization. Many times people find these two terms confusing. Some treat these two as the same. Actually, lemmatization is preferred over Stemming because lemmatization does morphological analysis of the words.
Applications of lemmatization are: 
 

Used in comprehensive retrieval systems like search engines.
Used in compact indexing
 
"""
"""
One major difference with stemming is that lemmatize takes a part of speech parameter, “pos” If not supplied, the default is “noun.”
Below is the implementation of lemmatization words using NLTK:
NLTK (Natural Language Toolkit) is a Python library used for natural language processing. One of its modules is the WordNet Lemmatizer, which can be used to perform lemmatization on words.

Lemmatization is the process of reducing a word to its base or dictionary form, known as the lemma. For example, the lemma of the word “cats” is “cat”, and the lemma of “running” is “run”.

Advantages of Lemmatization with NLTK:

Improves text analysis accuracy: Lemmatization helps in improving the accuracy of text analysis by reducing words to their base or dictionary form. This makes it easier to identify and analyze words that have similar meanings.
Reduces data size: Since lemmatization reduces words to their base form, it helps in reducing the data size of the text, which makes it easier to handle large datasets.
Better search results: Lemmatization helps in retrieving better search results since it reduces different forms of a word to a common base form, making it easier to match different forms of a word in the text.
Useful for feature extraction: Lemmatization can be useful in feature extraction tasks, where the aim is to extract meaningful features from text for machine learning tasks.
Disadvantages of Lemmatization with NLTK:

Requires prior knowledge: Lemmatization requires prior knowledge of the language and the rules governing the formation of words. If the rules for a specific language are not available, then the accuracy of the lemmatizer may be affected.
Time-consuming: Lemmatization can be time-consuming since it involves parsing the text and performing a lookup in a dictionary or a database of word forms.
Not suitable for real-time applications: Since lemmatization is time-consuming, it may not be suitable for real-time applications that require quick response times.
May lead to ambiguity: Lemmatization may lead to ambiguity, as a single word may have multiple meanings depending on the context in which it is used. In such cases, the lemmatizer may not be able to determine the correct meaning of the word.

"""
"""
Snowball Stemmer: It is a stemming algorithm which is also known as the Porter2 stemming algorithm as it is a better version of the Porter Stemmer since some issues of it were fixed in this stemmer.

First, let’s look at what is stemming-

Stemming: It is the process of reducing the word to its word stem that affixes to suffixes and prefixes or to roots of words known as a lemma. In simple words stemming is reducing a word to its base word or stem in such a way that the words of similar kind lie under a common stem. For example – The words care, cared and caring lie under the same stem ‘care’. Stemming is important in natural language processing(NLP).

Some few common rules of Snowball stemming are:


Few Rules:
ILY  -----> ILI
LY   -----> Nil
SS   -----> SS
S    -----> Nil
ED   -----> E,Nil
Nil means the suffix is replaced with nothing and is just removed.
There may be cases where these rules vary depending on the words. As in the case of the suffix ‘ed’ if the words are ‘cared’ and ‘bumped’ they will be stemmed as ‘care‘ and ‘bump‘. Hence, here in cared the suffix is considered as ‘d’ only and not ‘ed’. One more interesting thing is in the word ‘stemmed‘ it is replaced with the word ‘stem‘ and not ‘stemmed‘. Therefore, the suffix depends on the word.
Let’s see a few examples:-

Word           Stem
cared          care
university     univers
fairly         fair
easily         easili
singing        sing
sings          sing
sung           sung
singer         singer
sportingly     sport
Code: Python code implementation of Snowball Stemmer using NLTK library


import nltk
from nltk.stem.snowball import SnowballStemmer
 
#the stemmer requires a language parameter
snow_stemmer = SnowballStemmer(language='english')
 
#list of tokenized words
words = ['cared','university','fairly','easily','singing',
       'sings','sung','singer','sportingly']
 
#stem's of each word
stem_words = []
for w in words:
    x = snow_stemmer.stem(w)
    stem_words.append(x)
     
#print stemming results
for e1,e2 in zip(words,stem_words):
    print(e1+' ----> '+e2)
Output:

cared ----> care
university ----> univers
fairly ----> fair
easily ----> easili
singing ----> sing
sings ----> sing
sung ----> sung
singer ----> singer
sportingly ----> sport
You can also quickly check what stem would be returned for a given word or words using the snowball site. Under its demo section, you can easily see what this algorithm does for various different words.

Other Stemming Algorithms:

Porter Stemmer: This is an old stemming algorithm which was developed by Martin Porter in 1980. As compared to other algorithms it is a very gentle stemming algorithm.
Lancaster Stemmer: It is the most aggressive stemming algorithm. We can also add our own custom rules in this algorithm when we implement this using the NLTK package. Since it’s aggressive it can sometimes give strange stems as well.
There are other stemming algorithms as well.

Difference Between Porter Stemmer and Snowball Stemmer:

Snowball Stemmer is more aggressive than Porter Stemmer.
Some issues in Porter Stemmer were fixed in Snowball Stemmer.
There is only a little difference in the working of these two.
Words like ‘fairly‘ and ‘sportingly‘ were stemmed to ‘fair’ and ‘sport’ in the snowball stemmer but when you use the porter stemmer they are stemmed to ‘fairli‘ and ‘sportingli‘.
The difference between the two algorithms can be clearly seen in the way the word ‘Sportingly’ in stemmed by both. Clearly Snowball Stemmer stems it to a more accurate stem.
Drawbacks of Stemming:

Issues of over stemming and under stemming may lead to not so meaningful or inappropriate stems.
Stemming does not consider how the word is being used. For example – the word ‘saw‘ will be stemmed to ‘saw‘ itself but it won’t be considered whether the word is being used as a noun or a verb in the context. For this reason, Lemmatization is used as it keeps this fact in consideration and will return either ‘see’ or ‘saw’ depending on whether the word ‘saw’ was used as a verb or a noun.

"""
"""
nltk.tokenize.BlanklineTokenizer
classnltk.tokenize.BlanklineTokenizer[source]
Tokenize a string, treating any sequence of blank lines as a delimiter. Blank lines are defined as lines containing no characters, except for space or tab characters.
"""
"""
What does the word_tokenize do?
word_tokenize is a function in Python that splits a given sentence into words using the NLTK library.
 In Python, we can tokenize with the help of the Natural Language Toolkit ( NLTK ) library.
"""
"""
WordNet is a lexical database of semantic relations between words that links words into semantic relations including synonyms, hyponyms, and meronyms. The synonyms are grouped into synsets with short definitions and usage examples. It can thus be seen as a combination and extension of a dictionary and thesaurus
"""
"""
A word and its semantics (meanings, relations, and usage in various contexts) play a very important role in Natural Language Processing (NLP). A meaningful sentence is composed of meaningful words. Many of the NLP tasks, like text classification, sentiment analysis, and most important, WSD (word sense disambiguation), rely on these sentence and word semantics.
"""
"""
Word Sense
Words are ambiguous, which means the same word can be used differently depending on the context. For example, a 'bank' could be a river bank or a financial institution. These meanings and variety due to context are captured by sense (or word sense).

A sense (or word sense) is a discrete representation of one aspect of the meaning of a word.

Representation of Word Sense

There are many ways of mathematically defining or representing words in the form of embeddings like Word2Vec or GloVe, which can also capture some kind of meaning and relation between words defined by co-occurrences. But they fail to answer:** How to define the meaning of a word?**

Another way of capturing the senses is using thesauruses and giving a textual definition for each sense.

bank: (sloping land (especially the slope beside a body of water)) "they pulled the canoe up on the bank"; "he sat on the bank of the river and watched the currents"
bank: depository financial institution, bank, banking concern, banking company (a financial institution that accepts deposits and channels the money into lending activities) "he cashed a check at the bank"; "that bank holds the mortgage on my home."
An alternate way is to capture the semantic relationship between words (or senses) like car IS-A vehicle is a relation defined as 'car is a type of vehicle'.

Such definitions and semantic relations are captured by online tools like WordNet.

Semantic Relations
Synonymy The senses of two separate words are called synonyms if the meanings of these words are identical or similar. Example: center/middle, run/jog, etc.

Antonymy Antonyms are words with opposite meanings.

Example: dark/light, fast/slow etc.

Taxonomic Relations

Word senses can be related taxonomically so that they can be classified in certain categories. A word (or sense) is a hyponym of another word or sense if the one denotes a subclass of the other and is conversely called hypernym. For example, man is a hyponym of animal, and animal is a hypernym of man. Alternatively, this hyponym/hypernym can be defined as IS-A relationship 'Man IS-A animal'

Meronymy The 'part-whole' relationship is called Meronymy. A wheel is part of car.

Explore free masterclasses by our top instructors
View All
master class instructor
SOLID Principles Every developer must know
Pragy Agarwal
10 October 2023 | 7:30 PM
2.50 Hrs
Register with 1-Click
View Details
master class instructor
Data Science using Python
Suraaj Hasija
10 October 2023 | 7:30 PM
2.50 Hrs
Register with 1-Click
View Details
master class instructor
Low-Level Design of Payment apps
Naman Bhalla
12 October 2023 | 7:30 PM
2.50 Hrs
Register with 1-Click
View Details
2,35,262+ learners have attended these Masterclasses.
What is the WordNet?
Now that we have discussed some NLP terms, let's get back to WordNet. WordNet is a large lexical database of words, senses, and their semantic relations. This project was started by George A. Miller in the mid-1980s, and captures the word and their senses. In WordNet, the sense is defined by a set of synonyms, called synsets, that have a similar meaning or sense. This means WordNet represents words (or senses) as lists of the word senses that can be used to express the concept. Here is an example of a synset. Sense for the word 'fool' can be defined by the list of synonyms as {chump, jester, gull, fritter, dupe, fool around}

sample sysnet and various groups

It can also be seen that English WordNet consists of three separate databases, one each for nouns and verbs and a third for adjectives and adverbs.

Synset
A synset in WordNet is an interface that is a part of NLTK that can be used to look up words in WordNet. A Synset instance has groupings of words that are synonymous or words that express similar concepts. Some words have a singular Synset, and some have multiple. Here's an example:

from nltk.corpus import wordnet
syn = wordnet.synsets('hello')[0]

print ("Synset name : ", syn.name())

# definition of the word
print ("\nSynset meaning : ", syn.definition())

# list of phrases that use the word 'hello' in context
print ("\nSynset example : ", syn.examples())

Output:

Synset name :   hello.n.01

Synset meaning :  an expression of greeting

Synset example :  ['every morning they exchanged polite hellos']

Structure of WordNet
A synonym set (synset) is a group of words that all refer to the same notion in Wordnet. The structure of the wordnet is made up of words and synsets linked together by conceptual-semantic links.

As we read earlier, the structure of WordNet consists of words, senses, and Synsets. The image below, best describes the structure of WordNet.

structure-of-wordnet

How to use WordNet?
In this example, we are going to showcase the usage of NLTK to explore WordNet for synsets, meanings, and various semantic relationships.

WordNet Is Available as A Corpus in Nltk. Download the Word Net Corpus and Its Dependencies.
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

Find All the Sysnets of 'fool'
from nltk.corpus import wordnet as wn
wn.synsets('fool')

Output:

[Synset('fool.n.01'),
 Synset('chump.n.01'),
 Synset('jester.n.01'),
 Synset('fool.v.01'),
 Synset('fritter.v.01'),
 Synset('gull.v.02'),
 Synset('horse_around.v.01')]

Find only Verb Systems of 'fool.'
wn.synsets('fool', pos=wn.VERB)

Output:

[Synset('fool.v.01'),
 Synset('fritter.v.01'),
 Synset('gull.v.02'),
 Synset('horse_around.v.01')]


Creating a Class to Look up Words in WordNet
What is the WordNetTagger()? 
A very common function in NLP, is part-of-speech tagging, which means we tag every word to its part of speech, such as verbs, nouns, etc. 
WordNet provides us with this functionality with the WordNetTagger() function. Let's look at how it is used with a code example.

Now to use the WordNetTagger() for part of speech tagging, we must first create a class to look up words with WordNet.

from nltk.tag import SequentialBackoffTagger
from nltk.corpus import wordnet
from nltk.probability import FreqDist

class WordNetTagger(SequentialBackoffTagger):

'''
>>> wt = WordNetTagger()
>>> wt.tag(['food', 'is', 'awesome'])
[('food', 'NN'), ('is', 'VB'), ('awesome', 'JJ')]
'''

    def __init__(self, *args, **kwargs):

        SequentialBackoffTagger.__init__(self, *args, **kwargs)
        self.wordnet_tag_map = {
        'n': 'NN',
        's': 'JJ',
        'a': 'JJ',
        'r': 'RB',
        'v': 'VB'
        }

    def chooseTag(self, tokens, index, history):

    word = tokens[index]
    freq_d = FreqDist()

    for synset in wordnet.synsets(word):
        freq_d[synset.pos()] += 1

    return self.wordnet_tag_map.get(freq_d.max())

Now, this class that is created will return the count of the number of each part of the speech tag found in the Synsets for a word, and then the most common tag (treebank tag) will be the main tag given using internal mapping.

Using a Simple WordNetTagger()
We can now use the simple WordNetTagger().

from taggers import WordNetTagger
from nltk.corpus import treebank

# Initializing the tagger
def_tag = DefaultTagger('NN')

# initializing the training and testing sets
train = treebank.tagged_sents()[:3000]
test = treebank.tagged_sents()[3000:]

wn_tag = WordNetTagger()
res = wn_tag.evaluate(test)

print ("Accuracy of WordNetTagger : ", res)

Output:

Accuracy of WordNetTagger : 0.1791

We can improve this accuracy, read on!

WordNetTagger class at the end of an NgramTagger backoff chain
Here we initialize a backoff tagger, and use unigrams, bigrams as well as trigrams.

from taggers import WordNetTagger
from nltk.corpus import treebank
from tag_util import backoff_tagger
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

# Initializing
def_tag = DefaultTagger('NN')

# initializing training and testing set	
train = treebank.tagged_sents()[:3000]
test = treebank.tagged_sents()[3000:]

tagger = backoff_tagger(train, [UnigramTagger, BigramTagger, TrigramTagger], backoff=wn_tagger)

acc = tagger.evaluate(test)

print ("Accuracy is : ", acc)

Output:

Accuracy is : 0.8848

Finding Similarity Using WordNet
WordNet can also be used to find similarities between two words, of course, similarity here means semantic similarity. WordNet provides us with a function that helps compute similarity. A higher number in the result implies greater similarity.

To calculate the similarity between two words, we must first represent the words as synsets, and then make use of the wup_similarity() function.

from nltk.corpus import wordnet

syn_1 = wordnet.synsets('hello')[0]
syn_2 = wordnet.synsets('selling')[0]

print(syn_1.wup_similarity(syn_2))

Output

0.26666666666666666

Finding Entailments
What do we mean when we say the word "entailments"? Well, entailments essentially mean implications. For example, looking implies seeing, or buying implies choosing and paying. Now WordNet has entailment links between words. For example, a link between the word try (in the legal sense) and arrest exists, because in order to try someone, you have to arrest them. In NLP, this is called - Troponymy.

Conclusion
A word sense is the locus of word meaning; definitions and meaning relations are defined at the level of the word sense rather than word forms.
Relations between senses include synonymy, antonymy, meronymy, and taxonomic relations hyponymy and hypernymy.
WordNet is a large database of lexical relations for English, and WordNets exist for a variety of languages.
WordNet can help in Word Sense Disambiguation.
"""
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
