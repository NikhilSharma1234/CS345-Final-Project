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
    print("benign_mal_train")

def benign_mal_test(model, X_test):
    print("benign_mal_test")

def mal_train(model_name, X_train, y_train):
    print("mal_train")

def mal_test(model, X_test):
    print("mal_test")

def evaluate_hierarchical(benign_preds, mal_preds, y_test):
    print("evaluate_hierarchical")