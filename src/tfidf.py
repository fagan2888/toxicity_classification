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

import matplotlib.pyplot as plt

from cleartext import *

from nltk.corpus import stopwords 
from nltk.tokenize import TweetTokenizer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

#train = pd.read_csv('../data/train.csv')
# test = pd.read_csv('../data/test.csv')
# sub = pd.read_csv('../data/sample_submission.csv')


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

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

def computeCountDict(commentsWods):
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
        idfDict[word] = math.log(lenData / countDict[word])
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

def computeTFIDFVector(review):
      tfidfVector = [0.0] * len(wordDict)
     
      # For each unique word, if it is in the review, store its TF-IDF value.
      for i, word in enumerate(wordDict):
          if word in review:
              tfidfVector[i] = review[word]
      return tfidfVector

import math

def sigmoid(x):
    return (1 / (1 + np.exp(-x))) 

def degrau(x):
    if x < 0.5:
        return 0
    else:
        return 1

# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)):
        activation += weights[i + 1] * row[i]
    
    return sigmoid(activation)
        
    #return degrau(activation)

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights



train = pd.read_csv('../data/train.csv')

for i in range(0, 105, 5):
    a = i/100
    b = round(a + 0.05, 2)
    fix = 1000
    print(a, b)
    
    n = len(train[(train['target'] >= a) & (train['target'] < b)])
    
    remove = n - fix
    if remove > 0:
        drop_indices = np.random.choice(train[(train['target'] >= a) & (train['target'] < b)].index, remove, replace=False)
        train = train.drop(drop_indices)
        
#plt.figure(figsize=(12,6))
#plot = train.target.plot(kind='box')
plot = train.target.plot(kind='hist',bins=20)


comments = train[['comment_text', 'target']].copy()
comments['comment_text'] = comments.comment_text.apply(preprocess)
comments['comment_text'] = comments.comment_text.apply(lambda x: correct_spelling(x, mispell_dict))
comments['comment_text'] = comments.comment_text.apply(lambda x: clean_text(x))
comments['comment_text'] = comments.comment_text.apply(lambda x: clean_number(x)) #incas
commentsWords = comments.comment_text.apply(lambda x: cleanText(x))

tfDict = commentsWords.apply(lambda x: computeTFDict(x))

#Stores the review count dictionary
countDict = computeCountDict(commentsWords)
  
for key, value in list(countDict.items()):
    if value <= 10:
        countDict.pop(key)
#len(countDict)
#for key, value in list(countDict.items()):
#    if value >= 1000:
#        countDict.pop(key)

#Stores the idf dictionary
idfDict = computeIDFDict(countDict, len(commentsWords))

#Stores the TF-IDF dictionaries
tfidfDict = [computeTFIDFDict(comment, idfDict) for comment in tfDict]

# Create a list of unique words
wordDict = sorted(countDict.keys())


l_rate = 0.2
n_epoch = 10

# Estimate Perceptron weights using stochastic gradient descent
weights = [0.0 for i in range(len(wordDict)+1)]

div = [1000, 50000, 90000, 130000, 180000, 220000, 240000, 270000, 290000, 310000]
b = 1000
for a in div: 
    for epoch in range(n_epoch):
        sum_error = 0.0
        progress = 0
        for features, idx  in zip(tfidfDict[a:a+b], tfDict.index[a:a+b]):
            row = computeTFIDFVector(features)
            vv = degrau(comments['target'][idx])
            prediction = predict(row, weights)
            error = vv - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            #weights[0] = weights[0] + l_rate * error * prediction * (1 - prediction)
            for i in range(len(row)):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
                #weights[i + 1] = weights[i + 1] + l_rate * error * prediction * (1 - prediction) * row[i]
            
            printProgressBar(progress, len(tfidfDict[a:a+b]), prefix = 'Progress:', suffix = 'Complete', length = 50)
            progress += 1
            
        print('\n>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    
dataTrain = np.zeros((20442, 4326))
rspTrain = np.zeros((20442,1))

index = 0
for features, idx in zip(tfidfDict, tfDict.index):
    row = computeTFIDFVector(features)
    rsp = comments['target'][idx] 
    
    for r in range(len(row)):
        dataTrain[index, r] = row[r]
    
    rspTrain[index] = rsp
    
    index += 1
    
a = 1000
n_epoch = 100
weights = [0.0 for i in range(len(wordDict)+1)]

for epoch in range(n_epoch):
    sum_error = 0.0
    
    progress = 0
    
    for row, rsp  in zip(dataTrain[:a,:], rspTrain[:a,:]):
        prediction = predict(row, weights)
        error = rsp - prediction
        
        sum_error += error**2
        
        weights[0] = weights[0] + l_rate * error * prediction * (1 - prediction)
		
        for i in range(len(row)):
            
            weights[i + 1] = weights[i + 1] + l_rate * error * prediction * (1 - prediction) * row[i]
        
        progress += 1
            
        printProgressBar(progress, dataTrain[:a,:].shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
    print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


a = 28
#v = degrau(comments['target'][tfDict.index[a]])
v = comments['target'][tfDict.index[a]]
row = computeTFIDFVector(tfidfDict[a])
prediction = predict(row, weights)
print(v, prediction)


#tfidfVector = [computeTFIDFVector(review) for review in np.random.choice(tfidfDict, 1000)]


#sortedCount = sorted(wordsCount.items(), key = lambda kv:(kv[1], kv[0]))

#print(sorted(wordsCount.items(), key = lambda kv:(kv[1], kv[0])))

#print(sorted(countDict.items(), key = lambda kv:(kv[1], kv[0])))
