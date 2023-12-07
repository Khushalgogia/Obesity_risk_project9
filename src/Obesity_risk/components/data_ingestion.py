import sys
import os
import pandas as pd
from src.Obesity_risk.exception import customexception
from src.Obesity_risk.my_logger import logging
from src.Obesity_risk.components.data_transformation import datatransformation
from src.Obesity_risk.components.model_trainer import ModelTrainer
from src.Obesity_risk.components.model_trainer import model_trainer_config


from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class dataingestionconfig:
    train_data_path: str = os.path.join("artifects", 'train.csv')
    test_data_path: str = os.path.join("artifects", 'test.csv')
    raw_data_path: str = os.path.join("artifects", 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = dataingestionconfig()


    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method")
        try:
            df = pd.read_csv(r'C:\Users\khush\Python, 12-7\Practice\Github\Obesity_risk_Project8\notebooks\data\ObesityDataSet.csv')
            logging.info('Read the dataset as df')


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index= False, header= True)
            logging.info("Train test split initiated")

            train_set,test_set = train_test_split(df, test_size = 0.2, random_state= 101)

            train_set.to_csv(self.ingestion_config.train_data_path,index= False, header= True) 
            test_set.to_csv(self.ingestion_config.test_data_path,index= False, header= True)

            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.train_data_path
            )
            
        except Exception as e:
            raise customexception(e,sys)
            
if __name__ == '__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=datatransformation()
    train_arr,test_arr,_=data_transformation.initiate_pipeline(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_training(train_arr,test_arr))