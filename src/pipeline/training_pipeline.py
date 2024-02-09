from src.components.data_ingestion import data_ingestion
from src.logger import logging
from src.exception import CustomException



# Data ingestion
if __name__ == '__main__':
    try:
        obj = data_ingestion()
        logging.info(f"Data ingestion and converting to csv is completed")
    except CustomException as e:
        logging.info(f"Error in data ingestion: {e}")