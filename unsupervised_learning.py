# Author: Nikhil Sharma
# Date: 5/11/2023
# CS 345 - Cyber AI
# unsupervised_learning.py

from sklearn.cluster import KMeans
import numpy as np

# Generates warnings so ignoring them
import warnings
warnings.filterwarnings("ignore")

def unsup_binary_train(X_train, y_train):
    # Train trained KMeans model and fit the data
    return KMeans(n_clusters=2).fit(X_train, y_train)
    
def unsup_binary_test(model, X_test, y_test):
    # Intialize variables
    correct = 0
    total = 0

    # Get the predictions
    predictions = model.predict(X_test)

    # Iterate through predictions
    for i in range(len(predictions)):
        # Increment total
        total += 1
        # if they match
        if predictions[i] == y_test.iloc[i]:
            # increment correct
            correct += 1
    return(correct/total)

def unsup_multiclass_train(X_train, y_train, k):
    # Return kMeans with variable k of clusters
    return KMeans(n_clusters=k).fit(X_train, y_train)

def unsup_multiclass_test(model, X_test, y_test):
    # Initialize variables
    correct = 0
    total = 0

    # Get predictions
    predictions = model.predict(X_test)

    # Iterate through predictions
    for i in range(len(predictions)):
        # Increment total
        total += 1

        # If they match
        if predictions[i] == y_test.iloc[i]:
            # Increment correct
            correct += 1
    # Return accuracy
    return(correct/total)