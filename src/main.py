#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:43:28 2019

@author: willian
"""

import numpy as np
import pandas as pd
import string
import math

import matplotlib.pyplot as plt


from tfidf import *


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


data = pd.read_pickle('../data/data.pkl', )


tfDict = data.comment_text.apply(lambda x: computeTFDict(x))

#Stores the review count dictionary
countDict = computeCountDict(data.comment_text)
  
for key, value in list(countDict.items()):
    if value <= 1:
        countDict.pop(key)
        

#Stores the idf dictionary
idfDict = computeIDFDict(countDict, len(data.comment_text))

#Stores the TF-IDF dictionaries
tfidfDict = [computeTFIDFDict(comment, idfDict) for comment in tfDict]

# Create a list of unique words
wordDict = sorted(countDict.keys())

train = [computeTFIDFVector(review, wordDict) for review in tfidfDict]

l_rate = 0.8
n_epoch = 100

# Estimate Perceptron weights using stochastic gradient descent
#weights = [0.0 for i in range(len(wordDict)+1)]


for epoch in range(n_epoch):
    sum_error = 0.0
    
    progress = 0
    
    for row, target in zip(train, data.target):
        prediction = predict(row, weights)
        error = target - prediction
        
        sum_error += error**2
        
        weights[0] = weights[0] + l_rate * error
		
        for i in range(len(row)):
            
            weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        
        progress += 1
            
        printProgressBar(progress, len(data), prefix = 'Progress:', suffix = 'Complete', length = 50)
    print()
    print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


