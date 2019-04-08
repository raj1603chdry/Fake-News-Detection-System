"""
 # @author rajchoudhary
 # @email raj.choudhary1603@gmail.com
 # @create date 2019-03-12 10:44:15
 # @modify date 2019-04-09 01:38:28
 # @desc [File for preprocessing the "Liar, Liar Pants on Fire" dataset.]
"""

# Importing the libraries
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import STOPWORDS, WordCloud

warnings.filterwarnings('ignore')


# Helper functions
def read_tsv(file_path):
    """Function to read the dataset with .tsv extension.
    
    Parameters:
    -----------
    file_path: string
        Contains the path to the file to be read.

    Returns:
    --------
    dataset: pandas dataframe
        Contains the dataset read.
    """

    dataset = pd.read_csv(file_path, sep='\t', header=None)
    return dataset


def preprocess_dataset(dataset):
    """Function to select the required columns from the dataset and
    convert the multiclass labels to binary class labels.
    
    Parameters:
    -----------
    dataset: pandas dataframe
        The dataset whose contents are to be processed.
    """

    columns_to_select = [1, 2]
    dataset = dataset.iloc[:, columns_to_select]
    dataset.columns = ['label', 'news']

    # Converting the multiclass labels to binary labels
    labels_map = {
        'true': 'true',
        'mostly-true': 'true',
        'half-true': 'true',
        'false': 'false',
        'barely-true': 'false',
        'pants-fire': 'false'
    }
    dataset['label'] = dataset['label'].map(labels_map)

    return dataset


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


def show_dataset_stats(dataset, dataset_name):
    """Function that shows the size of the dataset alongwith
    5 sample columns.

    Parameters:
    -----------
    dataset: pandas dataframe
        Dataset whose stats are to be shown.
    dataset_name: string
        The name of the dataset.
    """
    print('Statistics of {}'.format(dataset_name))
    print('Shape: {}'.format(dataset.shape))
    print('Few samples from dataset:')
    print(dataset.sample(5))
    print()


def save_lowercase(dataset, file_path, file_name):
    """Function to convert all the string in the dataset into lowercase
    and save it in csv format.

    Parameters:
    -----------
    dataset: pandas dataframe
        Dataset to be saved in lowercase.
    file_path: string
        The path where the dataset is to be stored.
    file_name: string
        The name of the saved file.
    """
    dataset['news'] = dataset['news'].str.lower()
    complete_file_path_with_name = file_path+file_name
    dataset.to_csv(complete_file_path_with_name, index=False)


def show_label_distribution(dataset, dataset_name):
    """Function to show the label distribution in the dataset.

    Parameters:
    -----------
    dataset: pandas dataframe
        The dataset whose label distribution is to be shown.
    dataset_name: string
        The name of the dataset.
    """
    print('Label distribution {}:'.format(dataset_name))
    distribution = dataset['label'].value_counts(normalize=True).reset_index()
    distribution['label'] = distribution['label']*100
    for _, row in distribution.iterrows():
        print('{}\t{:.2f}%'.format(row['index'], row['label']))
    print()


def create_distribution(dataset, dataset_name):
    """Function to show the distribution of labels in the dataset and
    save the plot in the images folder with proper name.

    Parameters:
    -----------
    dataset: pandas dataframe
        Dataset whose distribution is to be displayed.
    dataset_name: string
        The name of the dataset.
    """
    if not os.path.isdir('../figures'):
        os.makedirs('../figures')
    sns.countplot(x='label', data=dataset)
    plt.title('Label distribution of '+dataset_name)
    plt.savefig('../figures/label_distribution_'+dataset_name)
    plt.show()


def check_dataset_quality(dataset, dataset_name):
    """Function to check the quality of the dataset i.e. if there are 
    missing values and cleaning them by removing those entries.

    Parameters:
    -----------
    dataset: pandas dataframe
        Dataset whose quality is to be checked.
    dataset_name: string 
        Name of the dataset.
    """
    print('Checking the quality of {}'.format(dataset_name))
    print(dataset.isnull().sum())
    print()
    number_of_nulls = dataset.isnull().sum().sum()
    if number_of_nulls > 0:
        dataset.dropna()


# File paths of the dataset to be read
train_path = '../datasets/train.tsv'
valid_path = '../datasets/valid.tsv'
test_path = '../datasets/test.tsv'

train_data = read_tsv(train_path)
valid_data = read_tsv(valid_path)
test_data = read_tsv(test_path)

# Preprocessing the datasets
train_data = preprocess_dataset(train_data)
valid_data = preprocess_dataset(valid_data)
test_data = preprocess_dataset(test_data)

# Path for saving the files
save_path = '../datasets/'

# Checking the quality of the dataset
check_dataset_quality(train_data, 'Train dataset')
check_dataset_quality(valid_data, 'Valid dataset')
check_dataset_quality(test_data, 'Test dataset')

# Saving the datasets in csv format
save_to_csv(train_data, save_path, 'train.csv')
save_to_csv(valid_data, save_path, 'valid.csv')
save_to_csv(test_data, save_path, 'test.csv')

# Displaying the stats of the datasets
show_dataset_stats(train_data, 'Train dataset')
show_dataset_stats(test_data, 'Test dataset')
show_dataset_stats(valid_data, 'Valid dataset')

# Display the label distribution of the datasets
show_label_distribution(train_data, 'Train dataset')
show_label_distribution(valid_data, 'Valid dataset')
show_label_distribution(test_data, 'Test dataset')

# Creating distributions of datasets
create_distribution(train_data, 'Train_dataset')
create_distribution(valid_data, 'Valid_dataset')
create_distribution(test_data, 'Test_dataset')

# Saving the datasets with all string in lowercase
save_lowercase(train_data, save_path, 'train_lower.csv')
save_lowercase(valid_data, save_path, 'valid_lower.csv')
save_lowercase(test_data, save_path, 'test_lower.csv')
