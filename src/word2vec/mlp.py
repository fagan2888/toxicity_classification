#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 09:42:05 2019

@author: willian
"""

import gensim

from nltk.corpus import stopwords 
from nltk.tokenize import TweetTokenizer

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import string

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
    
    words = [w for w in words if len(w) > 2 and len(w) < 16 and w != '']    
   
    return words


class MyTokenizer:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        transformed_X = []
        for document in X:
            transformed_X.append(np.array(cleanText(document)))
        return np.array(transformed_X)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = word2vec.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)
        
        return np.array([
            np.sum([self.word2vec[w] for w in words if w in self.word2vec.vocab]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)


# Data
data = pd.read_pickle('../../data/data2.pkl')
X = data.comment_text.values
y = data.target.values
del data

# Load word2vec model (trained on Google)
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True) 

mean_embedding_vectorizer = MeanEmbeddingVectorizer(model)
mean_embedded = mean_embedding_vectorizer.fit_transform(X)

clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(5,2,), max_iter=200, learning_rate_init=0.001, verbose=True)
prob = cross_val_predict(clf, mean_embedded, y, cv=StratifiedKFold(n_splits=10), method='predict_proba', verbose=2)

pred_indices = np.argmax(prob, axis=1)
classes = np.unique(y)
pred = classes[pred_indices]

cm = confusion_matrix(y, pred)
#skplt.plot_confusion_matrix(y, pred)

tpr = cm[1,1] / sum(cm[1,:])
fpr = cm[1,0] / sum(cm[:,0])
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
plt.plot(fpr, tpr, marker='.')


