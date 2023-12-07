import pandas as pd
import numpy as np
from src.Obesity_risk.exception import customexception
from src.Obesity_risk.my_logger import logging
import sys
import os

from src.Obesity_risk.utils.utils import saveobject, load_model

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifects', 'model.pkl')
            preprocessor_path = os.path.join('artifects', 'preprocessor.pkl')

            logging.info("Now will start predicting")

            model = load_model(model_path)
            preprocessor = load_model(preprocessor_path)

            transformed_data = preprocessor.transform(features)
            predicted_data = model.predict(transformed_data)

            return predicted_data
        except Exception as e:
            raise customexception(e,sys)
        
class Customdata:
    def __init__(self, gender, age, height, weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF,
                 TUE, CALC, MTRANS):
        self.gender = gender
        self.age = age
        self.height = height
        self.weight = weight
        self.family_history_with_overweight = family_history_with_overweight
        self.FAVC = FAVC
        self.FCVC = FCVC
        self.NCP = NCP
        self.CAEC = CAEC
        self.SMOKE = SMOKE
        self.CH2O = CH2O
        self.SCC = SCC
        self.FAF = FAF
        self.TUE = TUE
        self.CALC = CALC
        self.MTRANS = MTRANS

    def custom_convert_to_df(self):
        try:
           custom_dic =  {"Gender": [self.gender], "Age": [self.age], "Height": [self.height],
                          "Weight": [self.weight], "family_history_with_overweight": [self.family_history_with_overweight],
                            "FAVC": [self.FAVC], "FCVC": [self.FCVC], "NCP": [self.NCP], "CAEC": [self.CAEC],
                          "SMOKE": [self.SMOKE], "CH2O": [self.CH2O], "SCC": [self.SCC],
                          "FAF": [self.FAF], "TUE": [self.TUE], "CALC": [self.CALC], "MTRANS": [self.MTRANS]}
           return pd.DataFrame(custom_dic)
                          
                          
                          
        except Exception as e:
            customexception(e,sys)
    