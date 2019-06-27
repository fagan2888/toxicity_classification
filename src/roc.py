#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 11:07:33 2019

@author: willian
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


## GNB (0), LOG (1), MLP (1)
## Bag of Words (0), TF-IDF (1), word2vec (2)
## tpr (0), fpr (1)

roc = np.array([[[0.7962, 0.2852], [0.7076, 0.4345], [0.8043, 0.4413]],
[[0.7320, 0.2778], [0.7114, 0.2796], [0.7197, 0.3119]],
[[0.7145, 0.3246], [0.7104, 0.3309], [0.7294, 0.3096]]])


fig, ax = plt.subplots(figsize=(6, 6))

ax.plot([0, 1], [0, 1], linestyle='--', lw=2)

## Word Of Bag
# GNB
ax.scatter(roc[0, 0, 1], roc[0, 0, 0], marker='o', color='b', label='WB-GNB')
# LOG
ax.scatter(roc[1, 0, 1], roc[1, 0, 0], marker='s', color='b', label='WB-LOG')
# MLP
ax.scatter(roc[2, 0, 1], roc[2, 0, 0], marker='^', color='b', label='WB-MLP')


## TF-IDF
# GNB
ax.scatter(roc[0, 1, 1], roc[0, 1, 0], marker='o', color='g', label='T-GNB')
# LOG
ax.scatter(roc[1, 1, 1], roc[1, 1, 0], marker='s', color='g', label='T-LOG')
# MLP
ax.scatter(roc[2, 1, 1], roc[2, 1, 0], marker='^', color='g', label='T-MLP')


## WORD2VEC
# GNB
ax.scatter(roc[0, 2, 1], roc[0, 2, 0], marker='o', color='m', label='WV-GNB')
# LOG
ax.scatter(roc[1, 2, 1], roc[1, 2, 0], marker='s', color='m', label='WV-LOG')
# MLP
ax.scatter(roc[2, 2, 1], roc[2, 2, 0], marker='^', color='m', label='WV-MLP')


bag = mpatches.Patch(color='b', label='Bag of Words')
tfidf = mpatches.Patch(color='g', label='TF-IDF')
w2v = mpatches.Patch(color='m', label='Word2Vec')

first_legend = plt.legend(handles=[bag, tfidf, w2v], loc='lower right')
ax.add_artist(first_legend)

gnb = mlines.Line2D([], [], color='black', marker='o', ls='', label='Gaussian Naive Bayes')
log = mlines.Line2D([], [], color='black', marker='s', ls='', label='Logistic Regression')
mlp = mlines.Line2D([], [], color='black', marker='^', ls='', label='Multilayer Perceptron')

second_legend = plt.legend(handles=[gnb, log, mlp], loc='upper right')
ax.add_artist(second_legend)

ax.set_title('ROC')
ax.set_xlabel('TFP')
ax.set_ylabel('TVP')
ax.grid(linestyle='--')
fig.tight_layout()
fig.savefig('roc.eps')





