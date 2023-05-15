CS345-Final-Project
Design

The program is designed to analyze network traffic data. Using IDS (Intrusion detection system) evaluation dataset, we will be able to identify which data is benign or malicious to an extent. We apply different classification, regression, unsupervised learning, and feature selection models to do so.

The design consists of different files for the different ways we use the data to analyze it. We start off with the `helpers.py` file which consists of various helper functions that help get, clean and split the data from CSV files. We then apply multi-class classification models such as Decision Tree, KNN, Perceptron and Neural Networks in the `multiclass_classification.py` file. The next set of analysis we do is with feature selection by finding the minimum features needed to hit an accuracy and to find the most important features in a dataframe in the `feature_selection.py` file. We then apply the unsupervised learning model KNN in the `unsupervised_learning.py` file. Finally, we apply regression in the `regression.py` file.

Imports and functions

Global Variables and Import:
There are no global variables in this whole program.

```
import helpers as help
import multiclass_classification as mcc
import feature_selection as fs
import unsupervised_learning as ul
import regression as reg
```
The above code block represents the imports of the files in the test script to be able to run the functions from those files.

```
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
```
The above code block represents the classifiers being imported in multiple files for this project.

```
import numpy as np
import pandas as pd
```
Here are the two imports required to make and deal with various dataframes and calculations.

```
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
```
The above code block represents the regression models used in `regression.py`.

Functions
File: `helpers.py`
`load_data(filename)`
This function takes a filename and loads the data in the file into a Pandas Dataframe. The Dataframe is returned from the function.
`clean_data(df, flag)`
This function takes a Pandas Dataframe and either remove or replace all NaN values. Flag will take either the value “replace” or “remove.” If you replace the NaN values, they WILL be replaced by the MEAN value. This function should remove any columns of the data that are not numerical features. This function returns a cleaned Dataframe.
`split_data(df, label)`
This function takes a Pandas Dataframe and splits the Dataframe into training and testing data. Label is a string with one of the keys from the data (first row). This function splits the data into 80% for training and 20% for testing. I use the FIRST 80% of data to train and the remaining to test. This function returns four Dataframes: X_train, y_train, X_test, and y_test.
File: `multiclass_classification.py`
`direct_multiclass_train(model_name, X_train, y_train)`
This function takes the model_name (“dt”, “knn”, “perceptron”, “nn”) as input along with the training data (two Dataframes) and returns a trained model.
`direct_multiclass_test(model, X_test, y_test)`
This function takes a trained model and evaluates the model on the test data, returning an accuracy value.
`benign_mal_train(model_name, X_train, y_train)`
This function takes the model_name (“dt”, “knn”, “perceptron”, “nn”) as input along with the training data (two Dataframes) and returns a trained binary model that distinguishes between benign and malicious samples.
`benign_mal_test(model, X_test)`
This function takes a trained model and test data and returns a list of predictions (one for each test sample).
`mal_train(model_name, X_train, y_train)`
This function should takes the model_name (“dt”, “knn”, “perceptron”, “nn”) as input along with the training data (two Dataframes) and returns a trained multi-class model that distinguishes between different malicious samples.
`mal_test(model, X_test)`
This function takes a trained model and test data and returns a list of predictions (one for each test sample).
`evaluate_hierarchical(benign_preds, mal_preds, y_test)`
This function takes the list of benign predictions, malicious predictions and the test labels as input. The function should returns the accuracy of the predictions.
File: `feature_selection.py`
`find_min_features(model_name, X_train, y_train, X_test, y_test, accuracy)`
This function takes a model_name, training data, test data and a target accuracy as input and returns a list of feature names that produce the desired accuracy.
`find_important_features(X_train, y_train)`
Chose to use DecisionTreeClassifier for this. This function takes the training data as input and returns a list of feature names ranked from most important to least important.
File: `unsupervised_learning.py`
`unsup_binary_train(X_train, y_train)`
This function should apply K-means with K=2 to the training data and returns the trained model.
`unsup_binary_test(model, X_test, y_test)`
This function takes the trained K-means model and the test data and returns the accuracy.
`unsup_multiclass_train(X_train, y_train, k)`
This function applies K-means (K=of different attacks + 1 for benign) to the training data and returns the trained model.
`unsup_multiclass_test(model, X_test, y_test)`
This function takes the trained K-means model and the test data and returns the accuracy.
File: `regression.py`
`benign_regression_train(model_name, X_train, y_train)`
This function takes the model name, and the training data as input and returns a regression model for benign data.
`benign_regression_test(model, X_test, y_test)`
This function takes the model and test data as input. We use the test data to identify a threshold distance that would correctly classify the malicious data as malicious. Returns this threshold.
`benign_regression_evaluate(model, X_test, y_test, threshold)`
This function takes the model, test data and threshold as input and returns the accuracy for binary malicious traffic  identification.
Test Script

Run the program by running python3 test_script.py. The test script imports all the files and uses that to run all the functions.