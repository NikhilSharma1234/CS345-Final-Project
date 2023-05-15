# Author: Nikhil Sharma
# Date: 5/11/2023
# CS 345 - Cyber AI
# regression.py

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import numpy as np

def benign_regression_train(model_name, X_train, y_train):
    if model_name=="dt":
        DTR = DecisionTreeRegressor(random_state=0)
        DTR.fit(X_train, y_train)
        return DTR
    elif model_name=="knn":
        KNN = KNeighborsRegressor(n_neighbors=33)
        KNN = KNN.fit(X_train, y_train)
        return KNN
    elif model_name=="perceptron":
        perceptron = LinearRegression()
        perceptron = perceptron.fit(X_train, y_train)
        return perceptron
    elif model_name=="nn":
        nn = MLPRegressor(hidden_layer_sizes=(5, 2))
        nn = nn.fit(X_train, y_train)
        return nn

def benign_regression_test(model, X_test, y_test):
    preds = model.predict(X_test)
    dist = np.linalg.norm(preds-y_test)
    return dist

def benign_regression_evaluate(model, X_test, y_test, threshold):
    correct = 0
    total = 0
    predictions = model.predict(X_test)
    for i in range(len(predictions)):
        total += 1
        if predictions[i] == y_test.iloc[i] and predictions[i] <= threshold:
            correct += 1
    return(correct/total)