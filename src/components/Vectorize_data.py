import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.logger import logging
from src.exception import CustomException
from src.utils import create_directory
import pickle



# DATA VECTORIZATION
def tfidf_transform(text_data):
    try:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=df.index)
        return tfidf_df
    except Exception as e:
        logging.error(f"Error in tfidf_transform: {e}")


def count_vectors(text_data):
    try:
        count_vectorizer = CountVectorizer()
        count_matrix = count_vectorizer.fit_transform(text_data)
        count_df = pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out(), index=df.index)
        return count_df
    except Exception as e:
        logging.error(f"Error in count_vectors: {e}")


# STARTING THE VECTORIZATION
def start_vectorizer():
    try:
        logging.info("Starting vectorizer")
        pre_pros_path = "artifacts/preprocessing_files/train_set.csv"
        df = pd.read_csv(pre_pros_path)
        text_data = df['articles']
        tfidf_df = tfidf_transform(text_data)
        count_df = count_vectors(text_data)
        logging.info("Vectorizer finished")

        tfidf_df.to_csv("artifacts/vectorize_files/tfidf_data.csv")
        count_df.to_csv("artifacts/vectorize_files/Count_vectorize.csv")

        tfidf_df.to_pickle("artifacts/pickle_file/tfidf.pkl")
        count_df.to_pickle("artifacts/pickle_file/Count_Vectorizer.pkl")
        logging.info("Vectorizer has completed")
        return tfidf_df

    except Exception as e:
        logging.error(f"Error in start_vectorizer: {e}")

