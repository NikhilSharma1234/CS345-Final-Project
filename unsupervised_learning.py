# Author: Nikhil Sharma
# Date: 5/11/2023
# CS 345 - Cyber AI
# unsupervised_learning.py

from sklearn.cluster import KMeans
import numpy as np

def unsup_binary_train(X_train, y_train):
    return KMeans(n_clusters=2).fit(X_train, y_train)
    
def unsup_binary_test(model, X_test, y_test):
    correct = 0
    total = 0
    predictions = model.predict(X_test)
    for i in range(len(predictions)):
        total += 1
        if predictions[i] == y_test.iloc[i]:
            correct += 1
    return(correct/total)

def unsup_multiclass_train(X_train, y_train, k):
    return KMeans(n_clusters=k).fit(X_train, y_train)

def unsup_multiclass_test(model, X_test, y_test):
    correct = 0
    total = 0
    predictions = model.predict(X_test)
    for i in range(len(predictions)):
        total += 1
        if predictions[i] == y_test.iloc[i]:
            correct += 1
    return(correct/total)