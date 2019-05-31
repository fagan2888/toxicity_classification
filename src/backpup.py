#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:04:28 2019

@author: willian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:45:43 2019

@author: willian
"""

import numpy as np
import pandas as pd
import string
import math

from cleartext import *

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, TweetTokenizer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

train = pd.read_csv('../data/train.csv')
# test = pd.read_csv('../data/test.csv')
# sub = pd.read_csv('../data/sample_submission.csv')

def cleanText(text):
     
    # split into words
    tokens = TweetTokenizer().tokenize(text)

    # convert to lower case
    tokens = [w.lower() for w in tokens]
    
    # filter out stop words
    stopWords = set(stopwords.words('english'))
    stopWords.add('can\'t')
    words = [w for w in tokens if not w in stopWords]
    
    # remove punctuation from each word
    tr = str.maketrans("", "", string.punctuation + '\n 012345689')
    words = [w.translate(tr) for w in words]
    
    words = [w for w in words if len(w) > 2 and len(w) < 16 and w != '']    
    
    # punctuationWords = set(string.punctuation)
    # words = [w for w in words if not w in punctuationWords]
    
    # lemmatisation
    lemmatizer = WordNetLemmatizer() 
    lemmatized = [lemmatizer.lemmatize(word) for word in words]

    # stemming
    porter = SnowballStemmer('english')
    stemmed = [porter.stem(lemm) for lemm in lemmatized]
    
    return stemmed

def computeReviewTFDict(review):
    """ Returns a tf dictionary for each review whose keys are all 
    the unique words in the review and whose values are their 
    corresponding tf.
    """
    #Counts the number of times the word appears in review
    reviewTFDict = {}
    for word in review:
        if word in reviewTFDict:
            reviewTFDict[word] += 1
        else:
            reviewTFDict[word] = 1
    #Computes tf for each word           
    for word in reviewTFDict:
        reviewTFDict[word] = reviewTFDict[word] / len(review)
    return reviewTFDict

def computeCountDict():
    """ Returns a dictionary whose keys are all the unique words in
    the dataset and whose values count the number of reviews in which
    the word appears.
    """
    countDict = {}
    # Run through each review's tf dictionary and increment countDict's (word, doc) pair
    for review in tfDict:
        for word in review:
            if word in countDict:
                countDict[word] += 1
            else:
                countDict[word] = 1
    return countDict

def computeIDFDict():
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    idfDict = {}
    for word in countDict:
        idfDict[word] = math.log(len(data) / countDict[word])
    return idfDict

def computeReviewTFIDFDict(reviewTFDict):
    """ Returns a dictionary whose keys are all the unique words in the
    review and whose values are their corresponding tfidf.
    """
    reviewTFIDFDict = {}
    #For each word in the review, we multiply its tf and its idf.
    for word in reviewTFDict:
        if word in idfDict:
            reviewTFIDFDict[word] = reviewTFDict[word] * idfDict[word]
    return reviewTFIDFDict

def computeTFIDFVector(review):
      tfidfVector = [0.0] * len(wordDict)
     
      # For each unique word, if it is in the review, store its TF-IDF value.
      for i, word in enumerate(wordDict):
          if word in review:
              tfidfVector[i] = review[word]
      return tfidfVector

toxicComments = train['comment_text'][train['target'] > 0.5].apply(preprocess)
toxicComments = toxicComments.apply(lambda x: correct_spelling(x, mispell_dict))
toxicComments = toxicComments.apply(lambda x: clean_text(x))
toxicComments = toxicComments.apply(lambda x: clean_number(x)) #incas

toxicComments = train['comment_text'][train['target'] > 0.75]
toxicWords = toxicComments.apply(lambda x: cleanText(x))
#tfDict = toxicWords.apply(lambda x: computeReviewTFDict(x))

#Stores the review count dictionary
countDict = computeCountDict()
  
#for key, value in list(countDict.items()):
#    if value < 10:
#        countDict.pop(key)

#Stores the idf dictionary
#idfDict = computeIDFDict()

#Stores the TF-IDF dictionaries
#data = toxicComments
#tfidfDict = [computeReviewTFIDFDict(review) for review in tfDict]


# Create a list of unique words
#wordDict = sorted(countDict.keys())


#tfidfVector = [computeTFIDFVector(review) for review in tfidfDict]




#sortedCount = sorted(wordsCount.items(), key = lambda kv:(kv[1], kv[0]))

#print(sorted(wordsCount.items(), key = lambda kv:(kv[1], kv[0])))

#print(sorted(countDict.items(), key = lambda kv:(kv[1], kv[0])))
