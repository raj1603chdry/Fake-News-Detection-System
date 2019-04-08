"""
 # @author rajchoudhary
 # @email raj.choudhary1603@gmail.com
 # @create date 2019-03-29 20:44:56
 # @modify date 2019-04-09 01:41:15
 # @desc [Script to find polarity and subjectivity of the news statements.]
"""

# Importing the libraries
import os
import warnings

import pandas as pd
import numpy as np

from pattern.en import sentiment

warnings.filterwarnings('ignore')


def save_to_csv(dataset, file_path, file_name):
    """Function to save the dataset in csv format.

    Parameters:
    -----------
    dataset: pandas dataframe
        The dataset to be saved in the csv format.
    file_path: String
        The path where the dataset is to be stored.
    file_name: String
        The name of the saved file.
    """
    complete_file_path_with_name = file_path + file_name
    dataset.to_csv(complete_file_path_with_name, index=False)


def generate_polarity_subjectivity(dataset):
    """Function to generate the polarity and subjectivity of the news
    statements.

    Parameters:
    -----------
    dataset: pandas dataframe
        Dataset containing the news statements.
    
    Returns:
    --------
    df: pandas dataframe
        Contains the polarity and subjectivity of the news in the dataset.
    """
    df = dataset['news'].apply(lambda s: pd.Series({
        'polarity': round(sentiment(s)[0], 3),
        'subjectivity': round(sentiment(s)[1], 3)
    }))
    return df


# File paths of the dataset to be read
train_path = '../datasets/train.csv'
valid_path = '../datasets/valid.csv'
test_path = '../datasets/test.csv'

# Importing the datasets
train_data = pd.read_csv(train_path)
valid_data = pd.read_csv(valid_path)
test_data = pd.read_csv(test_path)

# Generating polarity and subjectivity
train_pol_sub = generate_polarity_subjectivity(train_data)
valid_pol_sub = generate_polarity_subjectivity(valid_data)
test_pol_sub = generate_polarity_subjectivity(test_data)

# Path for saving the files
save_path = '../datasets/'

# Saving the datasets in csv format
save_to_csv(train_pol_sub, save_path, 'train_pol_sub.csv')
save_to_csv(valid_pol_sub, save_path, 'valid_pol_sub.csv')
save_to_csv(test_pol_sub, save_path, 'test_pol_sub.csv')
