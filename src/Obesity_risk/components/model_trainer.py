import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.Obesity_risk.exception import customexception
from src.Obesity_risk.my_logger import logging
from src.Obesity_risk.utils.utils import saveobject, evaluate_model

@dataclass
class model_trainer_config:
    trained_model_file_path = os.path.join("artifects", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = model_trainer_config()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting and training and test input data")
            xTrain,yTrain,xTest,yTest = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )

            models = {
                "Decision Tree" : DecisionTreeClassifier(),
                "Random Forest" : RandomForestClassifier(),
                "Bagging"       : BaggingClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost"      : AdaBoostClassifier(),
                "KNN"           : KNeighborsClassifier(),
                "SVM"           : SVC()

            }
            logging.info("Now will train the model")
            model_report:dict = evaluate_model(xTrain = xTrain, yTrain = yTrain, xTest = xTest,yTest =  yTest, models = models)
            logging.info("Model Training done")
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(models.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise customexception(f"All models score are less than 60% with best score being {best_model_score}")
            logging.info('Best model score')


            saveobject(file_path= self.model_trainer_config.trained_model_file_path, 
                       obj = best_model)
            
            predicted = best_model.predict(xTest)

            accurate_score  = accuracy_score(yTest, predicted)
            logging.info(f"model report : {model_report}")
            return f'Best Model name : {best_model_name}\n Score : {accurate_score}'
        except Exception as e:
            raise customexception(e,sys)
