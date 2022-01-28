from flask import Flask, render_template, request,jsonify
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

app = Flask(__name__)
# model load


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return render_template('home.html')
    
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            age      = int(request.form['age'])  
            bmi      = float(request.form['bmi']) 
            children = int(request.form['children'])
            sex  = request.form['sex']      
            if sex == 'male':
                sex=0
            else:
                sex=1
            smoker   = request.form['smoker']
            if smoker == 'yes':
                smoker = 1
            else:
                smoker = 0

            region   = request.form['region'] 
            if region == 'northeast':
                region = [1, 0, 0]
            elif region == 'northwest':
                region = [0,0,1]
            elif region == 'southeast':
                region = [0, 1,0]
            else:
                region = [0, 0, 0]
    
            

            predict_input = [age, bmi, children, sex, smoker] + region
            predict_input = np.array(predict_input)
            predict_input = predict_input.reshape(1, -1)
            sclr = StandardScaler()
            predict_input = sclr.fit_transform(predict_input)
            model_load = pickle.load(open('insurance_premium.pkl', 'rb'))
            model_prediction = model_load.predict(predict_input)
            model_prediction = round(float(model_prediction), 2)
            

        except (ValueError, UnboundLocalError) as e:
             return render_template('incorrectinput.html')

    return render_template('predict.html', prediction = model_prediction)
@app.route('/report', methods=['POST'])
def report():
    if request.method == 'POST':
        return render_template('report.html')

@app.route('/training_detail', methods=['POST'])
def training_detail():
    if request.method == 'POST':
        return render_template('insurance premium prediction.html')




if __name__ == '__main__':
    app.run(port=6001, debug=True)