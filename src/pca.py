#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:59:17 2019

@author: willian
"""

import numpy as np

def normalize(x):
    n, m = x.shape    
    norm = np.zeros((n,m))
    
    for i in range(m):
        norm[:,i] = x[:,i] - np.mean(x[:,i])
        
    return norm
        

def covariance(x):
    n, m = x.shape
    cov = np.zeros((m, m))
    
    for i in range(m):
        for j in range(m):
            cov[i, j] = sum((x[:,i] - np.mean(x[:,i])) * (x[:,j] - np.mean(x[:, j]))) / (n - 1)
    
    return cov


A = np.array([[3, 4], [1, 2], [0, 4]])

covA = covariance(normalize(A))

eigA, eigV = np.linalg.eig(covA)

comps = eigV.transpose().dot(A.transpose()).transpose()


