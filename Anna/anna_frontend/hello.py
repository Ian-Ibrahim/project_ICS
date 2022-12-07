from flask import Flask, request, render_template
import numpy as np
import re
import sys 
import os
import base64
sys.path.append(os.path.abspath("../"))

from sklearn.preprocessing import LabelEncoder, StandardScaler
app = Flask(__name__)
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 1)
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
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)  
        # houseType=request.form.get("type")
        # print(type(houseType))
        # houseType=StandardScaler().fit_transform(float(houseType).reshape(-1, 1))
        # houseSubType=request.form.get("sub_type")
        # county=request.form.get("county")
        # locality=request.form.get("locality")
        # bedrooms=request.form.get("bedrooms")
        # bathrooms=request.form.get("bathrooms")
        # toilets=request.form.get("toilets")
        # parking=request.form.get("toilets")
        # furnished=request.form.get("furnished")
        # serviced=request.form.get("serviced")
        # shared=request.form.get("shared")
        # month=request.form.get("month")
        # year=request.form.get("year")
        # if furnished is None:
        #     furnished= int(0)
        # if furnished=="on":
        #     furnished=int(1)
        # if serviced is None:
        #     serviced= int(0)
        # if serviced=="on":
        #     serviced=int(1)
        # if shared is None:
        #     shared= int(0)
        # if shared=="on":
        #     shared=int(1)
        #     variables=[houseType,houseSubType,county,locality,bedrooms,bathrooms,toilets,
        # parking,furnished,serviced,shared,month,year]
        # houseType=StandardScaler().fit_transform(houseType.reshape(-1, 1))
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