#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 09:16:22 2019

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

from nltk.corpus import stopwords 
from nltk.tokenize import TweetTokenizer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer


# train = pd.read_csv('../data/train.csv')
# test = pd.read_csv('../data/test.csv')
# sub = pd.read_csv('../data/sample_submission.csv')

def cleanText(text):
     
    # split into words
    tokens = TweetTokenizer().tokenize(text)

    # convert to lower case
    tokens = [w.lower() for w in tokens]
    
    # filter out stop words
    stopWords = set(stopwords.words('english'))
    words = [w for w in tokens if not w in stopWords]
    
    # remove punctuation from each word
    tr = str.maketrans("", "", string.punctuation + '\n 012345689')
    words = [w.translate(tr) for w in words]
    
    # remove 
    words = [w for w in words if len(w) > 2 and len(w) < 16 and w != '']    
    
    # lemmatisation
    lemmatizer = WordNetLemmatizer() 
    lemmatized = [lemmatizer.lemmatize(word) for word in words]

    # stemming
    porter = SnowballStemmer('english')
    stemmed = [porter.stem(lemm) for lemm in lemmatized]
    
    return stemmed

def train_naive_bayes(training, classes):
    """Given a training dataset and the classes that categorize
    each observation, return V: a vocabulary of unique words,
    logprior: a list of P(c), and loglikelihood: a list of P(fi|c)s
    for each word
    """
    #Initialize D_c[ci]: a list of all documents of class i
    #E.g. D_c[1] is a list of [reviews, ratings] of class 1
    D_c = [[]] * len(classes)

    #Initialize n_c[ci]: number of documents of class i
    n_c = [None] * len(classes)

    #Initialize logprior[ci]: stores the prior probability for class i
    logprior = [None] * len(classes)

    #Initialize loglikelihood: loglikelihood[ci][wi] stores the likelihood probability for wi given class i
    loglikelihood = [None] * len(classes)
    
    #Partition documents into classes. D_c[0]: negative docs, D_c[1]: positive docs
    for obs in training.values:    #obs: a [review, rating] pair
        #D_c[0] = D_c[0] + [obs]
        #if rating >= 90, classify the review as positive
        if obs[1] > 0.5:
            D_c[1] = D_c[1] + [obs]    #Can also write as D_c[1] = D_c[1].append(obs)
        #else, classify review as negative
        elif obs[1] <= 0.5:
            D_c[0] = D_c[0] + [obs]
            
    V = set()
    for obs in training.values:
        for word in obs[0]:
            if word in V:
                continue
            else:
                V.add(word)
                
    V_size = len(V)
    
        #n_docs: total number of documents in training set
    n_docs = len(training.values)

    for ci in range(len(classes)):
        #Store n_c value for each class
        n_c[ci] = len(D_c[ci])
        
        #Compute P(c)
        logprior[ci] = np.log((n_c[ci] + 1)/ n_docs)

        #Counts total number of words in class c
        count_w_in_V = 0
        for d in D_c[ci]:
            count_w_in_V = count_w_in_V + len(d[0])
        denom = count_w_in_V + V_size

        dic = {}
        #Compute P(w|c)
        for wi in V:
            #Count number of times wi appears in D_c[ci]
            count_wi_in_D_c = 0
            for d in D_c[ci]:
                for word in d[0]:
                    if word == wi:
                        count_wi_in_D_c = count_wi_in_D_c + 1
            numer = count_wi_in_D_c + 1
            dic[wi] = np.log((numer) / (denom))
            
        loglikelihood[ci] = dic
        
    return V, logprior, loglikelihood

def test_naive_bayes(testdoc, logprior, loglikelihood, V):
    #Initialize logpost[ci]: stores the posterior probability for class ci
    logpost = [None] * len(classes)
    
    for ci in classes:
        sumloglikelihoods = 0
        for word in testdoc:
            if word in V:
                #This is sum represents log(P(w|c)) = log(P(w1|c)) + log(P(wn|c))
                sumloglikelihoods += loglikelihood[ci][word]
                
        
        #Computes P(c|d)
        logpost[ci] = logprior[ci] + sumloglikelihoods
        
    #Return the class that generated max ĉ
    return logpost.index(max(logpost))



toxicComments = train[['comment_text', 'target']][train['target'] > 0][:100]
toxicComments['comment_text'] = toxicComments['comment_text'].apply(lambda x: cleanText(x))
training = toxicComments 
classes = [0, 1]
testdoc = toxicComments.values[0][0]
V, logprior, loglikelihood = train_naive_bayes(toxicComments, classes)
rsp = test_naive_bayes(testdoc, logprior, loglikelihood, V)


