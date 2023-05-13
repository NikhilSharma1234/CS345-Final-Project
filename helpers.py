# Author: Nikhil Sharma
# Date: 5/11/2023
# CS 345 - Cyber AI
# helpers.py

import pandas as pd

# load_data function-params: filename
def load_data(filename):
    #  return the dataframe returned by read_csv with the file name as input
    return pd.read_csv(filename)

# clean_data function-params: df (the dataframe), flag (what we'd like to do with the nan values)
def clean_data(df, flag):
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

    # Clean the label coolumn and assign them assocaited numbers
    new_df['Label'] = new_df['Label'].replace(['Benign', 'FTP-BruteForce', 'SSH-Bruteforce'], [0, 1, 2])

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
    y_train = df[label]

    # form testing returnable variables
    X_test = df_20[features]
    y_test = df[label]

    return X_train, y_train, X_test, y_test