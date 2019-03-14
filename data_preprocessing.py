"""
 # @author rajchoudhary
 # @email raj.choudhary1603@gmail.com
 # @create date 2019-03-12 10:44:15
 # @modify date 2019-03-13 23:29:56
 # @desc [File for preprocessing the "Liar, Liar Pants on Fire" dataset.]
"""


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from wordcloud import WordCloud, STOPWORDS


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
    if not os.path.isdir('./figures'):
        os.makedirs('figures')
    sns.countplot(x='label', data=dataset)
    plt.title('Label distribution of '+dataset_name)
    plt.savefig('./figures/label_distribution_'+dataset_name)
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


def plot_save_wordcloud(dataset, title):
    """Function to plot the wordcloud of the entries of the dataset and
    save it in the figures folder.

    Parameters:
    -----------
    dataset: pandas dataframe
        The dataset containing the sentences.
    title: string
        Title of wordcloud.
    """
    if not os.path.isdir('./figures'):
        os.makedirs('figures')

    text = dataset['news'].values
    wordcloud = WordCloud(width=3000, height=2000, background_color='white',
                          stopwords=STOPWORDS).generate(str(text))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title('{}'.format(title))
    title = title.replace(' ', '_')
    plt.savefig('./figures/word_cloud_'+title)
    plt.show()


def create_word_cloud(dataset, dataset_name):
    """Function to create the wordcloud of the complete dataset as well as
    for only the true labels as well as for false labels of the dataset.

    Parameters:
    -----------
    dataset: pandas dataframe
        The dataset whose wordcloud is to be formed.
    dataset_name: string
        The name of the dataset.
    """

    title = 'The wordcloud of the complete '+dataset_name
    plot_save_wordcloud(dataset, title)

    # Plotting the wordcloud for the true labels only
    true_dataset = dataset[dataset['label'] == 'true']
    title = 'The wordcloud of the true labels of '+dataset_name
    plot_save_wordcloud(true_dataset, title)

    # Plotting the wordcloud for the false labels only
    false_dataset = dataset[dataset['label'] == 'false']
    title = 'The wordcloud of the false labels of '+dataset_name
    plot_save_wordcloud(false_dataset, title)


# File paths of the dataset to be read
train_path = './datasets/train.tsv'
valid_path = './datasets/valid.tsv'
test_path = './datasets/test.tsv'

train_data = read_tsv(train_path)
valid_data = read_tsv(valid_path)
test_data = read_tsv(test_path)

# Preprocessing the datasets
train_data = preprocess_dataset(train_data)
valid_data = preprocess_dataset(valid_data)
test_data = preprocess_dataset(test_data)

# Path for saving the files
save_path = './datasets/'

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

# Creating distributions of datasets
create_distribution(train_data, 'Train_dataset')
create_distribution(valid_data, 'Valid_dataset')
create_distribution(test_data, 'Test_dataset')

# Creating the wordclouds of the datasets
create_word_cloud(train_data, 'Train dataset')
create_word_cloud(valid_data, 'Valid dataset')
create_word_cloud(test_data, 'Test dataset')