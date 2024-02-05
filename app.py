from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.Obesity_risk.pipelines.prediction_pipeline import PredictPipeline, Customdata

application = Flask(__name__)

app = application

#Route for homepage

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoints():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = Customdata(
            gender= request.form.get('Gender'),
            age= request.form.get('Age'),
            height= request.form.get('Height'),
            weight= request.form.get('Weight'),
            family_history_with_overweight= request.form.get('family_history_with_overweight'),
            FAVC= request.form.get('FAVC'),
            FCVC= request.form.get('FCVC'),
            NCP= request.form.get('NCP'),
            CAEC= request.form.get('CAEC'),
            SMOKE= request.form.get('SMOKE'),
            CH2O= request.form.get('CH2O'),
            SCC= request.form.get('SCC'),
            FAF= request.form.get('FAF'),
            TUE= request.form.get('TUE'),
            CALC= request.form.get('CALC'),
            MTRANS= request.form.get('MTRANS'))
        
        df_pred = data.custom_convert_to_df()

        pred_pipeline = PredictPipeline()
        result = pred_pipeline.predict(df_pred)

        return render_template('home.html', results = result[0])

    

if __name__=="__main__":
    app.run(debug= True)    
