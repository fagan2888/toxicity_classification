#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:45:43 2019

@author: willian
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from cleartext import *

# Balanced database
def balanceDatabase(train):
    for i in range(0, 105, 5):
        a = i/100
        b = round(a + 0.05, 2)
        fix = 1000
        
        n = len(train[(train['target'] >= a) & (train['target'] < b)])
        
        remove = n - fix
        if remove > 0:
            drop_indices = np.random.choice(train[(train['target'] >= a) & (train['target'] < b)].index, remove, replace=False)
            train = train.drop(drop_indices)
            
    plt.figure(figsize=(12,6))
    train.target.plot(kind='box')
    plt.figure(figsize=(12,6))    
    train.target.plot(kind='hist',bins=20)

# Two class -1 and 1
def classTarget(target):
    for index, items in target.iteritems():
        if items < 0.5:
            target[index] = -1
        else:
            target[index] = 1
            
    pos = target[target > 0].shape[0]
    neg = target.shape[0] - pos
    
    plt.figure(figsize=(12,6))
    plt.bar([1,-1], [pos, neg])

# Token
def tokenText(comments):    
    comments = comments.apply(preprocess)
    comments = comments.apply(lambda x: correct_spelling(x, mispell_dict))
    comments = comments.apply(lambda x: clean_text(x))
    comments = comments.apply(lambda x: clean_number(x)) #incas
    commentsWords = comments.apply(lambda x: cleanText(x))    
    
    return commentsWords


train = pd.read_csv('../data/train.csv')
balanceDatabase(train)
comments = train['comment_text'].copy()
target = train['target'].copy()
classTarget(target)
commentsWords = tokenText(comments)
data = pd.concat([commentsWords, target], axis=1)
data.reset_index(drop=True)
data.to_csv('../data/data.csv')

