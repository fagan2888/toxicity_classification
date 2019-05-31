#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:43:47 2019

@author: willian
"""

l_rate = 0.8
n_epoch = 100

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
            #vv = degrau(comments['target'][idx])
            vv = comments['target'][idx]
            prediction = predict(row, weights)
            #print(prediction)
            #error = vv - prediction
            error = vv * np.log(prediction) + (1 - vv) * np.log(1 - prediction)
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            #weights[0] = weights[0] + l_rate * error * prediction * (1 - prediction)
            for i in range(len(row)):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
                #weights[i + 1] = weights[i + 1] + l_rate * error * prediction * (1 - prediction) * row[i]
            
            #printProgressBar(progress, len(tfidfDict[a:a+b]), prefix = 'Progress:', suffix = 'Complete', length = 50)
            progress += 1
            
        print('\n>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    