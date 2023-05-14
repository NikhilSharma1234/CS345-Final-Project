# Author: Nikhil Sharma
# Date: 5/11/2023
# CS 345 - Cyber AI
# helpers.py

import pandas as pd
import numpy as np

# load_data function-params: filename
def load_data(filename):
    #  return the dataframe returned by read_csv with the file name as input
    return pd.read_csv(filename)

# clean_data function-params: df (the dataframe), flag (what we'd like to do with the nan values)
def clean_data(df, flag):
    # Map the Label column values to use later
    d = {'Benign': 0, 'FTP-BruteForce': 1, 'SSH-Bruteforce': 2}

    # Clean the label coolumn and assign them associated numbers
    df['Label'] = df['Label'].map(d)

    # if, flag is remove
    if(flag=="remove"):
        # remove the rows with nan values
        new_df = df.dropna()

    # else,
    else:
        # Get the mean values of the columns
        df_means = df.mean(0)

        # Make a copy
        new_df = df.copy()

        # iterate through the rows ro find those nan values to replace with mean
        for item in new_df:
            if item in df_means:
                new_df[item].fillna(df_means[item], inplace=True)

    # Remove other non numerical data
    new_df = new_df.select_dtypes(['number'])

    # Get rid of all infinite values
    new_df = new_df[np.isfinite(new_df).all(1)]

    # Return the dataframe
    return new_df

def split_data(df, label):
    # Get out list of features minus the label column
    features = list(df)[:-1]

    # Randomly sample 80%
    df_80 = df.sample(frac = 0.8)

    # Get the rest of the 20% of the dataframe
    df_20 = df.drop(df_80.index)

    # form training returnable variables
    X_train = df_80[features]
    y_train = df_80[label]

    # form testing returnable variables
    X_test = df_20[features]
    y_test = df_80[label]

    return X_train, y_train, X_test, y_test