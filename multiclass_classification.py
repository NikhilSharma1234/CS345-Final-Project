# Author: Nikhil Sharma
# Date: 5/11/2023
# CS 345 - Cyber AI
# multiclass_classification.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

def direct_multiclass_train(model_name, X_train, y_train):
    # Use classifier model depending on input model_name
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

def direct_multiclass_test(model, X_test, y_test):
    # Initalize variables
    correct = 0
    total = 0
    # Get predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    for i in range(len(predictions)):
        total += 1
        if predictions[i] == y_test.iloc[i]:
            correct += 1
    
    # Return accuracy
    return(correct/total)

def benign_mal_train(model_name, X_train, y_train):
    # Need binary data so map anything malicious as 1
    d = {0: 0, 1: 1, 2: 1}

    # Map the y_train
    y_train = y_train.map(d)

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

def benign_mal_test(model, X_test):
    # Return predictions of the model
    return model.predict(X_test)

def mal_train(model_name, X_train, y_train):
    # Get lsit of indicies where the attribute is equal to 0
    benign_index_list = y_train[y_train == 0].index

    # Modify y_train to remove all rows with occurences of 0
    y_train = y_train[y_train != 0]

    # Remove X_train rows with list of indices from earlier
    X_train = X_train.drop(benign_index_list)

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

def mal_test(model, X_test):
    # Return predictions of the model
    return model.predict(X_test)

def evaluate_hierarchical(benign_preds, mal_preds, y_test):
    # Intialize Variables
    correct = 0
    total = 0

    # Run loop to get accuracy for benign
    for i in range(len(benign_preds)):
        total += 1
        if benign_preds[i] == y_test.iloc[i]:
            correct += 1
    # Run loop to get accuracy for mal_preds
    for i in range(len(mal_preds)):
        total += 1
        if mal_preds[i] == y_test.iloc[i]:
            correct += 1
    # Return the total accuracy
    return(correct/total)