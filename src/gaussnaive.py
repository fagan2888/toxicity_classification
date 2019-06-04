#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:45:30 2019

@author: willian
"""

import numpy as np
import pandas as pd


def train_naive_bayes(training):
    """Given a training dataset and the classes that categorize
    each observation, return V: a vocabulary of unique words,
    logprior: a list of P(c), and loglikelihood: a list of P(fi|c)s
    for each word
    """
    #Initialize D_c[ci]: a list of all documents of class i
    #E.g. D_c[1] is a list of [reviews, ratings] of class 1
    #Partition documents into classes. D_c[0]: negative docs, D_c[1]: positive docs
    possitiveClass = training[training['target.1'] == 1].values[:,:-1]
    negativeClass = training[training['target.1'] == 0].values[:,:-1]
    
    logprior = np.zeros((2,))
    logprior[1] = np.log(len(possitiveClass)/ len(training))
    logprior[0] = np.log(len(negativeClass)/ len(training))
    
    
    media = np.zeros((2, len(possitiveClass[0])))
    var = np.zeros((2, len(possitiveClass[0])))
    
    
    media[1] = np.mean(possitiveClass, axis=0)
    var[1] = np.var(possitiveClass, axis=0)
    
    
    media[0] = np.mean(negativeClass, axis=0)
    var[0] = np.var(negativeClass, axis=0)
    
        
    return logprior, media, var


#data2 = pd.read_csv('../data/tfidfdata.csv')

#data2 = data2.drop(np.random.choice(data2.index, int(len(data2)/2), replace=False))

train_naive_bayes(data2)