# Author: Nikhil Sharma
# Date: 5/11/2023
# CS 345 - Cyber AI
# multiclass_classification.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

def direct_multiclass_train(model_name, X_train, y_train):
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

def direct_multiclass_test(model, X_test, y_test):
    correct = 0
    total = 0
    predictions = model.predict(X_test)
    for i in range(len(predictions)):
        total += 1
        if predictions[i] == y_test.iloc[i]:
            correct += 1
    return(correct/total)

def benign_mal_train(model_name, X_train, y_train):
    d = {0: 0, 1: 1, 2: 1}
    y_train = y_train.map(d)
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

def benign_mal_test(model, X_test):
    return model.predict(X_test)

def mal_train(model_name, X_train, y_train):
    benign_index_list = y_train[y_train == 0].index
    y_train = y_train[y_train != 0]
    X_train = X_train.drop(benign_index_list)
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

def mal_test(model, X_test):
    return model.predict(X_test)

def evaluate_hierarchical(benign_preds, mal_preds, y_test):
    correct = 0
    total = 0
    for i in range(len(benign_preds)):
        total += 1
        if benign_preds[i] == y_test.iloc[i]:
            correct += 1
    for i in range(len(mal_preds)):
        total += 1
        if mal_preds[i] == y_test.iloc[i]:
            correct += 1
    return(correct/total)