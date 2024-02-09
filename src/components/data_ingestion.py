import os
import sys
import tarfile
from src.logger import logging
from src.exception import CustomException




# EXTRACTING THE RAW DATA FROM TAR FILES
def extract_tar_files(tar_file_path:str)->str:
    try:
        logging.info(f"Extracting the raw data from the tar file: {tar_file_path}")
        extract_to = os.path.join('artifacts','raw_file')
        with tarfile.open(tar_file_path,'r') as tar:
            tar.extractall(path=extract_to)
        logging.info(f"Extracted the raw data from the tar file")
    except Cust as e:
        logging.info(f"Error extracting the raw data from the tar file{e}")











def data_ingestion():
    try:
        tar_file_path = "artifacts/tar_file/RACE.tar.gz"
        extracted_dir_path = extract_tar_files(tar_file_path)
    except Exception as e:
        logging.info(f"Error in data ingestion: {e}")






