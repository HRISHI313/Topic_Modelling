from src.components.data_ingestion import data_ingestion
from src.components.data_transformation import start_transformation
from src.components.Vectorize_data import start_vectorizer
from src.logger import logging
from src.exception import CustomException



# TRAINING PIPELINE
if __name__ == '__main__':
    try:
        # logging.info('Data ingestion has been started')
        # obj = data_ingestion()
        # logging.info(f"Data ingestion and converting to csv is completed")
        #
        # logging.info("Data transformation has been initiated")
        # df1, df2 = start_transformation()
        # logging.info("Data transformation has been completed")

        logging.info("Vectorizer has been started")
        df1 = start_vectorizer()
        logging.info("Vectorizer has been completed")


    except CustomException as e:
        logging.error(f"Error in training_pipeline: {e}")