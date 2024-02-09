import os
import sys
import json
import csv
import tarfile
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split

# EXTRACTING THE RAW DATA FROM TAR FILES
def extract_tar_files(tar_file_path:str):
    try:
        logging.info(f"Extracting the raw data from the tar file: {tar_file_path}")
        extract_to = os.path.join('artifacts','raw_file')
        with tarfile.open(tar_file_path,'r') as tar:
            tar.extractall(path=extract_to)
        logging.info(f"Extracted the raw data from the tar file")
    except Cust as e:
        logging.info(f"Error extracting the raw data from the tar file{e}")


# CONVERTING THE TEXT FILE INTO CSV FILES
def read_json_file(file_path):
    logging.info("Reading the JSON files")
    try:
        with open(file_path, 'r') as file:
            content = json.load(file)
            return content.get("article", "")
    except Exception as e:
        logging.info(f"Error in data ingestion: {e}")

def process_folders(root_folder_paths, csv_file_path):
    try:
        logging.info(f"Processing the folders: {root_folder_paths}")
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for root_folder_path in root_folder_paths:
                for root, dirs, files in os.walk(root_folder_path):
                    for filename in files:
                        if filename.endswith('.txt'):
                            file_path = os.path.join(root, filename)
                            article_text = read_json_file(file_path)
                            writer.writerow([article_text])

    except Exception as e:
        logging.info(f"Error in data ingestion: {e}")



# READING THE CSV FILE AND SPLITTING IT INTO TRAIN AND TEST DATA

def read_csv_file(csv_path: str) -> pd.DataFrame:
    try:
        logging.info(f"Reading the csv file: {csv_path}")
        df = pd.read_csv(csv_path, header=None)
        columns = ['articles']
        df.columns = columns
        logging.info("Adding the column in the CSV")
        logging.info(df.head(3))
        return df

    except Exception as e:
        logging.info(f"Error in data ingestion: {e}")



def data_splitting(dataframe):
    try:
        logging.info(f"Splitting the dataframe into 80% and 20%")
        train_set, test_set = train_test_split(dataframe, test_size=0.2, random_state=42)

        logging.info("Setting up the train and test path")
        train_data = os.path.join('artifacts','train_set.csv')
        test_data = os.path.join('artifacts','test_set.csv')

        logging.info("Putting the data in the train data and test data")
        train_set.to_csv(train_data,index=False)
        test_set.to_csv(test_data,index=False)

        return train_set, test_set

    except Exception as e:
        logging.info(f"Error in data ingestion: {e}")



def data_ingestion():
    try:
        tar_file_path = "artifacts/tar_file/RACE.tar.gz"
        root_folder_paths = ['artifacts/raw_file/RACE/dev', 'artifacts/raw_file/RACE/test', 'artifacts/raw_file/RACE/train']
        csv_file_path = 'artifacts/ingested_data.csv'
        # extracted_dir_path = extract_tar_files(tar_file_path)
        # processed_data = process_folders(root_folder_paths, csv_file_path)
        data_ingest = data_splitting(read_csv_file(csv_file_path))

    except Exception as e:
        logging.info(f"Error in data ingestion: {e}")






