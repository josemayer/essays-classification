import logging
import os
import sys
from typing import Tuple, List

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('build_dataset')


class Corpus:
    """
    This class creates and reads the training, development, and testing sets.
    """

    def build_corpus(self, single_df: bool = False) -> None:
        """
        Build splits for training, development, and testing of the corpus
        :return:
        """

        if single_df:
            df_train, _ = self.read_corpus()

            training, development, testing = split_stratified_into_train_val_test(df_train,
                                                                                  stratify_colname='score',
                                                                                  random_state=42,
                                                                                  frac_train=0.7,
                                                                                  frac_val=0.15,
                                                                                  frac_test=0.15)
            self.save_split('train-single', training)
            self.save_split('dev-single', development)
            self.save_split('test-single', development)
        else:
            training, development, testing = split_stratified_into_train_val_test_with_two_dfs(self.read_corpus(),
                                                                                  stratify_colname='score',
                                                                                  random_state=42)
            self.save_split('train', training)
            self.save_split('dev', development)
            self.save_split('test', testing)
    
    @staticmethod
    def read_splits(single_df: bool = False) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Reads the splits of the corpus
        :param single_df: bool indicating if split is relatively to a single dataframe
        :return: training, development, and testing
        """
        path = 'data/splits'

        if single_df:
            train_file, dev_file, test_file = 'train-single.csv', 'dev-single.csv', 'test-single.csv'
        else:
            train_file, dev_file, test_file = 'train.csv', 'dev.csv', 'test.csv'

        training = pd.read_csv(os.path.join(path, train_file), converters={'essay': eval, 'competence': eval})
        development = pd.read_csv(os.path.join(path, dev_file), converters={'essay': eval, 'competence': eval})
        testing = pd.read_csv(os.path.join(path, test_file), converters={'essay': eval, 'competence': eval})
        return training, development, testing
    
    @staticmethod
    def read_corpus() -> Tuple[DataFrame, DataFrame]:
        path = 'data/'

        train = pd.read_csv(os.path.join(path, 'train.csv'), converters={'essay': eval, 'grades': eval})
        test = pd.read_csv(os.path.join(path, 'test.csv'), converters={'essay': eval, 'grades': eval})

        return format_corpus(train), format_corpus(test)

    @staticmethod
    def save_split(name: str, df_input: DataFrame) -> None:
        """
        Save the splits of the corpus as a csv file
        :param name: name of the split
        :param df_input: content of the splits as a data frame
        :return:
        """
        df_input.to_csv('data/splits/'+name+'.csv', index=False, header=True)
        logger.info(name + '.csv saved in data/splits/')

def format_corpus(corpus: DataFrame) -> DataFrame:
    relevant_labels = ['id_prompt', 'title', 'essay', 'grades']
    new_corpus = corpus[relevant_labels]
    new_corpus = new_corpus.rename(columns={"id_prompt": "prompt", "grades": "competence"})

    new_corpus['competence'] = new_corpus['competence'].map(lambda x : x[:-1])
    new_corpus['score'] = new_corpus['competence'].map(lambda x : sum(x))

    return new_corpus

def split_stratified_into_train_val_test_with_two_dfs(dfs_input, stratify_colname='y', frac_train=0.8, frac_val=0.2,
                                         random_state=None) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    dfs_input : Tuple of pandas dataframes
        Input dataframes of train and test to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
        The ratios with which the dataframe will be split into train and val 
        data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    """

    if frac_train + frac_val != 1.0:
        raise ValueError('fractions %f and %f do not add up to 1.0' % (frac_train, frac_val))

    if len(dfs_input) != 2:
        raise ValueError('tuple of train and test dataframes do not have size of 2')

    train, test = dfs_input

    if stratify_colname not in train.columns or stratify_colname not in test.columns:
        raise ValueError('%s is not a column of at least one of test and train dataframes' % stratify_colname)

    X_train = train
    Y_train = train[[stratify_colname]]

    X_test = test

    # Split original train dataframe into train and val dataframes.
    df_train, df_val, _, _ = train_test_split(X_train, Y_train, test_size=(1.0 - frac_train), random_state=random_state)

    # Define original test dataframe as main test dataframe
    df_test = X_test
 
    assert len(train) == len(df_train) + len(df_val)
    assert len(test) == len(df_test)

    return df_train, df_val, df_test

def split_stratified_into_train_val_test(df_input, stratify_colname='y', frac_train=0.8, frac_val=0.1, frac_test=0.1,
                                         random_state=None) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    """

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f and %f do not add up to 1.0' % (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column of input dataframe' % stratify_colname)

    X = df_input
    Y = df_input[[stratify_colname]]

    # Split dataframe into train and temp dataframes
    df_train, df_temp, _, y_temp = train_test_split(X, Y, test_size=(1.0 - frac_train), random_state=random_state)

    # Split temp dataframe into test and val dataframes 
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, _, _ = train_test_split(df_temp, y_temp, test_size=relative_frac_test, 
                                             random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

if __name__ == '__main__':
    single_df = False
    if len(sys.argv) > 1 and sys.argv[1] == 'single':
        single_df = True

    Corpus().build_corpus(single_df = single_df)
    train, valid, test = Corpus().read_splits(single_df = single_df)
    print(test.head())
    print(train.head())
    
    print(len(train))
    print(len(valid))
    print(len(test))
