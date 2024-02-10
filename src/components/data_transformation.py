import os
import sys
import pandas as pd
import string
import re
from src.logger import logging
from src.exception import CustomException
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer



# CONVERTING THE TEXT INTO LOWERCASE
def lower_case(train_data: str, test_data: str):
    try:
        logging.info(f"opening the {train_data} and {test_data}")
        df1 = pd.read_csv(train_data)
        df2 = pd.read_csv(test_data)

        logging.info(f"Converting the text into lower")
        df1['articles'] = df1['articles'].str.lower()
        df2['articles'] = df2['articles'].str.lower()

        return df1, df2

    except CustomException as e:
        logging.info(f"Error in lower_case: {e}")



# REMOVAL OF PUNCTUATION
def removal_punctuation(df1: pd.DataFrame, df2: pd.DataFrame):
    try:
        logging.info(f"Removing punctuation")
        df1['articles'] = df1['articles'].str.replace('[^\w\s]','')
        df2['articles'] = df2['articles'].str.replace('[^\w\s]','')

        logging.info("Punctuation has been removed")
        return df1, df2

    except CustomException as e:
        logging.info(f"Error in removal_punctuation: {e}")



# REMOVAL OF STOPWORDS
def removal_stopwords(df1: pd.DataFrame, df2: pd.DataFrame):
    try:
        logging.info(f"Removing the stopwords")
        stop_words = set(stopwords.words('english'))
        df1['articles'] = df1['articles'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
        df2['articles'] = df2['articles'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

        logging.info("Stopwords have been removed")
        return df1, df2

    except CustomException as e:
        logging.info(f"Error in removal_stopwords: {e}")



# REMOVAL OF FREQURNT WORDS
def removal_frequent_words(df1: pd.DataFrame, df2: pd.DataFrame):
    try:
        word_count1 = Counter()
        for text1 in df1['articles']:
            for word1 in text1.split():
                word_count1[word1] += 1

        word_count2 = Counter()
        for text2 in df2['articles']:
            for word2 in text2.split():
                word_count2[word2] += 1

        # Determine frequent words for both dataframes
        frequent_words_df1 = set(word for (word, wc) in word_count1.most_common(3))
        frequent_words_df2 = set(word for (word, wc) in word_count2.most_common(3))

        # Remove frequent words from articles
        df1['articles'] = df1['articles'].apply(lambda x: " ".join(word for word in x.split() if word not in frequent_words_df1))
        df2['articles'] = df2['articles'].apply(lambda x: " ".join(word for word in x.split() if word not in frequent_words_df2))

        logging.info("Frequent words have been removed")
        return df1, df2

    except CustomException as e:
        logging.error(f"Error in removal_frequent_words: {e}")


# REMOVAL OF SPECIAL CHARACTER
def removal_special_characters(df1: pd.DataFrame, df2: pd.DataFrame):
    try:
        logging.info(f"Removing the special characters")
        df1['articles'] = df1['articles'].apply(lambda x: " ".join(word for word in x.split() if word not in string.punctuation))
        df2['articles'] = df2['articles'].apply(lambda x: " ".join(word for word in x.split() if word not in string.punctuation))

        logging.info("Special characters have been removed")
        return df1, df2

    except CustomException as e:
        logging.info(f"Error in removal_special_characters: {e}")



# STEMMING
def stemming(df1: pd.DataFrame, df2: pd.DataFrame):
    try:
        logging.info(f"Stemming the words")
        ps = PorterStemmer()
        df1['articles'] = df1['articles'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))
        df2['articles'] = df2['articles'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))

        logging.info("Stemming has been applied")
        return df1, df2

    except CustomException as e:
        logging.info(f"Error in stemming: {e}")



# LEMMA
























