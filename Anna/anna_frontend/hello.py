from flask import Flask, request, render_template
import numpy as np
import re
import sys 
import os
import base64
import pickle
sys.path.append(os.path.abspath("../"))
import model
from model import ANNRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
app = Flask(__name__)
def ValuePredictor(to_predict_list):
    
    loaded_model = pickle.load(open("../models/model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
 
@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict",methods=['GET','POST'])
def predict():
    result=0;
    month=5
    
    if request.method =="POST":
        to_predict_list=list() 
        bedrooms=request.form.get("bedrooms")
        to_predict_list.append(bedrooms)
        bathrooms=request.form.get("bathrooms")
        to_predict_list.append(bathrooms)
        toilets=request.form.get("toilets")
        to_predict_list.append(toilets)
        furnished=request.form.get("furnished")
        serviced=request.form.get("serviced")
        shared=request.form.get("shared")
        to_predict_list.append(furnished)
        to_predict_list.append(serviced)
        to_predict_list.append(shared)
        parking=request.form.get("parking")
        to_predict_list.append(parking)
        category=1
        to_predict_list.append(category)
        houseType=request.form.get("type")
        to_predict_list.append(houseType)
        houseSubType=request.form.get("sub_type")
        to_predict_list.append(houseSubType)
        county=request.form.get("county")
        to_predict_list.append(county)
        locality=request.form.get("locality")
        to_predict_list.append(locality)
        year=request.form.get("year")
        to_predict_list.append(year)
        month=request.form.get("month")
        to_predict_list.append(month)
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list) 
    return render_template('predict.html',prediction=result)

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/login")
def login():
    return render_template('authentication/login.html')

@app.route("/register")
def register():
    return render_template('authentication/register.html')

# handle 404 error
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404
if __name__ == "__main__":
    app.run(debug=True)