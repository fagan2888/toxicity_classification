#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 18:48:01 2019

@author: willian
"""     

import pandas as pd
import numpy as np

def computeTFDict(comment):
    """ Returns a tf dictionary for each comment whose keys are all 
    the unique words in the review and whose values are their 
    corresponding tf.
    """
    #Counts the number of times the word appears in review
    commentTFDict = {}
    for word in comment:
        if word in commentTFDict:
            commentTFDict[word] += 1
        else:
            commentTFDict[word] = 1
    
    #Computes tf for each word           
    for word in commentTFDict:
        commentTFDict[word] = commentTFDict[word] / len(comment)
    return commentTFDict

def computeCountDict(commentsWords):
    """ Returns a dictionary whose keys are all the unique words in
    the dataset and whose values count the number of reviews in which
    the word appears.
    """
    countDict = {}
    # Run through each review's tf dictionary and increment countDict's (word, doc) pair
    for comment in commentsWords:
        for word in comment:
            if word in countDict:
                countDict[word] += 1
            else:
                countDict[word] = 1
    return countDict

def computeIDFDict(countDict, lenData):
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    idfDict = {}
    for word in countDict:
        idfDict[word] = np.log(lenData / countDict[word])
    return idfDict

def computeTFIDFDict(commentTFDict, idfDict):
    """ Returns a dictionary whose keys are all the unique words in the
    review and whose values are their corresponding tfidf.
    """
    commentTFIDFDict = {}
    #For each word in the review, we multiply its tf and its idf.
    for word in commentTFDict:
        if word in idfDict:
            commentTFIDFDict[word] = commentTFDict[word] * idfDict[word]

    return commentTFIDFDict

def computeTFIDFVector(review, wordDict):
      tfidfVector = [0.0] * len(wordDict)
     
      # For each unique word, if it is in the review, store its TF-IDF value.
      for i, word in enumerate(wordDict):
          if word in review:
              tfidfVector[i] = review[word]
      return tfidfVector


data = pd.read_pickle('../data/data.pkl')

#Calculates the tf of terms
tfDict = data.comment_text.apply(lambda x: computeTFDict(x))

#Stores the review count dictionary
countDict = computeCountDict(data.comment_text)        
#Remove words with frequency <= 1
for key, value in list(countDict.items()):
    if value <= 1:
        countDict.pop(key)
        
    
#Stores the idf dictionary
idfDict = computeIDFDict(countDict, len(data.comment_text))

#Stores the TF-IDF dictionaries
tfidfDict = [computeTFIDFDict(comment, idfDict) for comment in tfDict]

# Create a list of unique words
wordDict = sorted(countDict.keys())

train = np.zeros((len(tfidfDict), len(wordDict) + 1))

index = 0
for review, target in zip(tfidfDict, data.target):
    a = computeTFIDFVector(review, wordDict)
    a.append(target)
    train[index, :] = a
    index += 1
    
wordDict.append('target')
traindata = pd.DataFrame(train, columns=wordDict)

traindata.to_csv('../data/tfidfdata.csv', index=False)
