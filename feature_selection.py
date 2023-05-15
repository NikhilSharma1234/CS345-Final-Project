# Author: Nikhil Sharma
# Date: 5/11/2023
# CS 345 - Cyber AI
# feature_selection.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

def find_min_features(model_name, X_train, y_train, X_test, y_test, accuracy):
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
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X_train, y_train)
    importanceSci = dtree.feature_importances_
    importanceDec = []
    index = 0
    for num in importanceSci:
        importanceDec.append(["{:.10f}".format(num), index])
        index += 1
    importanceDec = sorted(importanceDec, key=lambda x:x[0], reverse=True)
    order = []
    for element in importanceDec:
        order.append(element[1])
    return [val for (_, val) in sorted(zip(order, X_train.columns), key=lambda x:x[1])]
