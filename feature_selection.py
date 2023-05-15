# Author: Nikhil Sharma
# Date: 5/11/2023
# CS 345 - Cyber AI
# feature_selection.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

def find_min_features(model_name, X_train, y_train, X_test, y_test, accuracy):
    # DID NOT FINISH
    # Run classifier depending on input
    # Simple fitting and returning of model
    if model_name=="dt":
        dtree = DecisionTreeClassifier()
        dtree = dtree.fit(X_train, y_train)
        return dtree
    elif model_name=="knn":
        KNN = KNeighborsClassifier(n_neighbors=3)
        KNN = KNN.fit(X_train, y_train)
        return KNN
    elif model_name=="perceptron":
        perceptron = Perceptron()
        perceptron = perceptron.fit(X_train, y_train)
        return perceptron
    elif model_name=="nn":
        nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        nn = nn.fit(X_train, y_train)
        return nn

def find_important_features(X_train, y_train):
    # Chose to use DecisionTreeClassifer to find important features
    # Train and fit data
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X_train, y_train)

    # Get the importance values (scientific notation)
    importanceSci = dtree.feature_importances_

    # Initalize empty list and variable
    importanceDec = []
    index = 0

    # Convert to decimal by iterating through list
    for num in importanceSci:
        # To the tenth decimal and append to importanceDec
        importanceDec.append(["{:.10f}".format(num), index])
        index += 1
    
    # sort importance Dec according to the importance value
    importanceDec = sorted(importanceDec, key=lambda x:x[0], reverse=True)

    # Initialize variable and append order of the features to it
    order = []
    for element in importanceDec:
        order.append(element[1])
    
    # Return the mapped feature names according to the order list from above
    return [val for (_, val) in sorted(zip(order, X_train.columns), key=lambda x:x[1])]
