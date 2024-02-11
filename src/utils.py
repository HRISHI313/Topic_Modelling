import os
import sys
from src.logger import logging
from src.exception import CustomException




# CREATING DIRECTORIES:
def create_directory(path:str):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Directory [{path}] has been created")
        else:
            logging.info(f"Directory [{path}] already exists")
    except CustomException as e:
        logging.error(f"Error in create_directory: {e}")