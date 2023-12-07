from dataclasses import dataclass
import sys
import os
from src.Obesity_risk.my_logger import logging
from src.Obesity_risk.exception import customexception
from src.Obesity_risk.utils.utils import saveobject


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer
import pickle

@dataclass
class datatransformationconfig:
    preprocessor_ob_file_path = os.path.join('artifects', "preprocessor.pkl")

class datatransformation:
    def __init__(self):
        self.datatransformationconfig = datatransformationconfig()

    def get_data_transformer_obj(self):
        try:
            all_columns = ['Gender', 'family_history_with_overweight', 'FAVC',  'SMOKE', 'SCC']
            num_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

            logging.info(f"Numerical columns : {num_columns}")
            logging.info(f"All the columns except numerical columns : {all_columns}")


            def MTRANS_map_Transformer(df):
                MTRANS_map = {'Public_Transportation' : 'Not Walking', 'Automobile' : 'Not Walking', 'Motorbike' : 'Not Walking', 'Bike' : 'Not Walking',"Walking" : 'Walking' }

                df['MTRANS'] = df['MTRANS'].map(MTRANS_map)
                encoder = {'Walking' : 1, 'Not Walking': 0 }
                df['MTRANS'] = df['MTRANS'].map(encoder)
                return df[['MTRANS']]

            def CAEC_CALC_categories_map(df):
                CAEC_CALC_categories= {'no' : 0,'Sometimes' : 1, 'Frequently': 2 ,'Always':3}
                
                df['CAEC'] = df['CAEC'].map(CAEC_CALC_categories)
                df['CALC'] = df['CALC'].map(CAEC_CALC_categories)
                return df[['CAEC','CALC']]
            logging.info('Preprocessing started')
            
            preprocessor1 = ColumnTransformer(
                transformers = [
                    ('MTRANS Transformer',FunctionTransformer(MTRANS_map_Transformer), ['MTRANS'] ),
                    ('CAEC_CALC_categories_transformer', FunctionTransformer(CAEC_CALC_categories_map),['CAEC', 'CALC']),
                    ('OrdinalEncoder', OrdinalEncoder(),all_columns ),
                    ('StandardScaler', Normalizer(), num_columns)
                    
                ],remainder = 'passthrough'
                                            )

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor1),
                
                # Other steps in your pipeline, like model training, etc.
                                ])
            logging.info("Pipeline to be returned")

            return pipeline

            #logging.info("Transformation Done")

            


        except Exception as e:
            raise customexception(e,sys)
    
    def initiate_pipeline(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            logging.info("Obtaining pipeline object")

            preprocessing_Obj = self.get_data_transformer_obj()

            target_column = "NObeyesdad"
            input_feature_train = train_df.drop(columns = target_column)
            target_feature_train = train_df[target_column]

            input_feature_test = test_df.drop(columns = target_column)
            target_feature_test = test_df[target_column]

            logging.info("Applying preprocessing object on training and test dataframe")
            
            input_feature_arr_train = preprocessing_Obj.fit_transform(input_feature_train)
            input_feature_arr_test = preprocessing_Obj.fit_transform(input_feature_test)

            train_arr = np.c_[input_feature_arr_train,np.array(target_feature_train)]
            test_arr = np.c_[input_feature_arr_test,np.array(target_feature_test)]

            logging.info("Now saving preprocessing object in pkl file")

            saveobject(
                file_path = self.datatransformationconfig.preprocessor_ob_file_path ,
                obj= preprocessing_Obj

            )
            logging.info("Data transformation done")

            return (
                train_arr, 
                test_arr,
                self.datatransformationconfig.preprocessor_ob_file_path
            )

            


        except Exception as e:
            raise customexception(e,sys)