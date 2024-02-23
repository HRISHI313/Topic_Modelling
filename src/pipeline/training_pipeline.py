from src.components.data_ingestion import data_ingestion
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import data_transformation
# from src.components.data_vectorization import tfidf



# Data ingestion
if __name__ == '__main__':
    try:
        # obj = data_ingestion()
        logging.info(f"Data ingestion and converting to csv is completed")
        logging.info(f"Data transformation has been initiated")
        # df1, df2 = data_transformation()
        logging.info(f"Data transformation has been completed and pickle file has been generated")
        # logging.info("Data Vectoriztion has been started")
        # df3 = tfidf("artifacts/train_set.csv")
    except CustomException as e:
        logging.info(f"Error in data ingestion: {e}")