#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:25:08 2019

@author: willian
"""


from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
#from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd



# Data
data = pd.read_pickle('../../data/data2.pkl')
X = data.comment_text.values
y = data.target.values
del data


## Word of Bags    
count_vectorizer = CountVectorizer(analyzer='word', tokenizer=TweetTokenizer().tokenize, stop_words=stopwords.words('english'), max_features=10000)    
bag_of_words = count_vectorizer.fit_transform(X)

#clf = GaussianNB()
clf = MultinomialNB()
#clf = ComplementNB()
prob = cross_val_predict(clf, bag_of_words.toarray(), y, cv=StratifiedKFold(n_splits=10), method='predict_proba', verbose=2)

pred_indices = np.argmax(prob, axis=1)
classes = np.unique(y)
pred = classes[pred_indices]

cm = confusion_matrix(y, pred)
#skplt.plot_confusion_matrix(y, pred)

tpr = cm[1,1] / sum(cm[1,:])
fpr = cm[1,0] / sum(cm[:,0])
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
plt.plot(fpr, tpr, marker='.')

