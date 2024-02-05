import sys
import numpy as np
import pandas as pd
import os
import pickle
import dill

from src.Obesity_risk.exception import customexception
from sklearn.metrics import accuracy_score

def saveobject(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok= True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise customexception(e,sys)
    
def evaluate_model(xTrain,yTrain,xTest,yTest, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(xTrain,yTrain)
            train_pred = model.predict(xTrain)
            test_pred = model.predict(xTest)
            train_accscore = accuracy_score(train_pred,yTrain)
            test_accscore = accuracy_score(test_pred,yTest)
            

            report[list(models.keys())[i]] = test_accscore
        return report
    except Exception as e:
        raise customexception(e,sys)
    

def load_model(file_obj):
    try:
        with open(file_obj, 'rb') as file:
            return pickle.load(file)

    except Exception as e:
        raise customexception(e,sys)