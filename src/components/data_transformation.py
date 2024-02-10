import os
import sys
import pandas as pd
import string
from src.logger import logging
from src.exception import CustomException



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























