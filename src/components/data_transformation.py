import os
import sys
import pandas as pd
import re
from src.logger import logging
from src.exception import CustomException
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src.utils import create_directory
import pickle


# CONVERTING THE TEXT INTO LOWER.
def to_lower(train_data: str, test_data: str):
    try:
        logging.info("Converting into lower has been initiated")
        df1 = pd.read_csv(train_data)
        df2 = pd.read_csv(test_data)
        logging.info("Reading the train data and test data")
        df1['articles'] = df1['articles'].str.lower()
        df2['articles'] = df2['articles'].str.lower()
        logging.info("Converted into lower string")
        return df1, df2
    except CustomException as e:
        logging.info(f"Error converting the text into lower {e}")


# REMOVAL OF STOPWORDS
def remove_stopwords(df1: pd.DataFrame, df2: pd.DataFrame):
    try:
        logging.info("Removing of stopwords")
        stop_words = set(stopwords.words('english'))
        df1['articles'] = df1['articles'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
        df2['articles'] = df2['articles'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
        logging.info("Stopwords have been removed")
        return df1, df2
    except CustomException as e:
        logging.info(f"Error removing the stopwords {e}")



# APPLYING TOKENIZATION
def tokenize(df1: pd.DataFrame, df2: pd.DataFrame):
    try:
        logging.info("Tokenization has been initiated")
        df1['articles'] = df1['articles'].apply(lambda x: " ".join(word_tokenize(x)))
        df2['articles'] = df2['articles'].apply(lambda x: " ".join(word_tokenize(x)))
        logging.info("Tokenization has been completed")
        return df1, df2
    except CustomException as e:
        logging.info(f"Error in tokenization: {e}")


# IMPLEMENTING STEMMING
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


# IMPLEMENTING LEMMATIZATION
def lemma(df1: pd.DataFrame, df2: pd.DataFrame):
    try:
        lemmatizer = WordNetLemmatizer()
        df1['articles'] = df1['articles'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
        df2['articles'] = df2['articles'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

        logging.info("Lemmatization has been applied")
        return df1, df2

    except CustomException as e:
        logging.info(f"Error in lemmatization: {e}")



# REMOVAL OF URL'S.
def remove_urls(text):
    try:
        # logging.info("Removing URLs")
        # Define the regular expression pattern to match URLs
        url_pattern = r'http\S+|www\S+|https\S+'
        cleaned_text = re.sub(url_pattern, '', text)
        return cleaned_text
    except CustomException as e:
        logging.error(f"Error in remove_urls: {e}")

def remove_urls_from_column(df1: pd.DataFrame, df2: pd.DataFrame):
    try:
        logging.info("Removing URLs from column 'articles'")
        df1['articles'] = df1['articles'].apply(remove_urls)
        df2['articles'] = df2['articles'].apply(remove_urls)
        logging.info("URLs have been removed")
        return df1, df2
    except CustomException as e:
        logging.error(f"Error in remove_urls_from_column: {e}")



# REMOVING NON-ALPHANUMERIC CHARACTER
def remove_non_alphanumeric(df1: pd.DataFrame, df2: pd.DataFrame):
    try:
        logging.info("Removing non-alphanumeric characters")
        df1['articles'] = df1['articles'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        df2['articles'] = df2['articles'].apply(lambda x: re.sub(r'[^\w\s]', '', x))


        logging.info("Non-alphanumeric characters have been removed")
        return df1, df2
    except CustomException as e:
        logging.info(f"Error in remove_non_alphanumeric: {e}")



# REMOVING NUMERICS
def remove_numerics(df1: pd.DataFrame, df2: pd.DataFrame):
    try:
        logging.info("Removing numerics")
        df1['articles'] = df1['articles'].apply(lambda x: re.sub(r'\d+', '', x))
        df2['articles'] = df2['articles'].apply(lambda x: re.sub(r'\d+', '', x))

        logging.info("Numerics have been removed")
        return df1, df2
    except CustomException as e:
        logging.info(f"Error in remove_numerics: {e}")





def start_transformation():
    try:
        train_path = r"artifacts/train_set.csv"
        test_path = r"artifacts/test_set.csv"
        logging.info(f"Data transformation has been initiated")
        df1, df2 = to_lower(train_path, test_path)
        df1, df2 = remove_stopwords(df1, df2)
        df1, df2 = tokenize(df1, df2)
        df1, df2 = stemming(df1, df2)
        df1, df2 = lemma(df1, df2)
        df1, df2 = remove_urls_from_column(df1, df2)
        df1, df2 = remove_non_alphanumeric(df1, df2)
        df1, df2 = remove_numerics(df1, df2)

        create_directory('artifacts/preprocessing_files')
        df1.to_csv('artifacts/preprocessing_files/train_set.csv', index=False)
        df2.to_csv('artifacts/preprocessing_files/test_set.csv', index=False)

        df1.to_pickle('artifacts/pickle_file/data_transform.pkl')

        return df1, df2
    except CustomException as e:
        logging.info(f"Error in start_transformation: {e}")









