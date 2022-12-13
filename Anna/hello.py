from flask import Flask, request, render_template
import numpy as np
import re
import sys 
import os
import base64
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
bedrooms=0
class ANNRegressor(BaseEstimator, RegressorMixin):
    # Constructor to instantiate default or user-defined values
    def __init__(self, in_features=12, num_hidden=1, num_neurons=36, epochs=50, 
                    batch_norm=False, early_stopping=True, verbose=1):
        self.in_features = in_features
        self.num_hidden = num_hidden
        self.num_neurons = num_neurons
        self.batch_norm = batch_norm
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.verbose = verbose
        
        # Build the ANN
        self.model = ANNRegressor.build_model(self.in_features, self.num_hidden, self.num_neurons, self.batch_norm)
        
    @staticmethod
    def build_model(in_features, num_hidden, num_neurons, batch_norm):
        model = Sequential()
        
        # Input layer
        model.add(Dense(num_neurons, input_shape=(in_features,), activation='relu'))

        # Add hidden layers to model
        if (num_hidden > 1):
            for i in range(num_hidden - 1):
                model.add(Dense(num_neurons, activation='relu'))
                if(batch_norm):
                    model.add(BatchNormalization())

        # Output layer
        model.add(Dense(1))
        
        return model
        
    def fit(self, X, Y):
        # Split into training and validating sets
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=1/3)
        
        # Specifies callbacks list
        callbacks = [
            ModelCheckpoint('models/annmodel.weights.hdf5', save_best_only=True, verbose=self.verbose)
            
        ]
        
        # Use early stopping to stop training when validation error reaches minimum
        if(self.early_stopping):
            callbacks.append(EarlyStopping(monitor='val_loss', patience=10, verbose=self.verbose))
        
        # Compile the model then train
        adam = Adam(learning_rate=0.001)
        self.model.compile(optimizer=adam, loss='mse')
        self.model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=self.epochs, 
                       callbacks=callbacks, verbose=self.verbose)
        
        model_json = self.model.to_json()
        with open("models/annmodel.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save('models/ann_housing.h5')
        
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions

app = Flask(__name__)
loaded_model = pickle.load(open("models/annmodel.pkl", "rb"))
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    result = loaded_model.predict(to_predict)
    print(result)
    return result[0]
 
@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict",methods=['GET','POST'])
def predict():
    result=0;
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
        pre_covid=0;
        to_predict_list.append(pre_covid)
        category=0
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
        formresult = request.form
        to_predict_list = list(map(int, to_predict_list))
        Rentalresult = ValuePredictor(to_predict_list) 
        result=Rentalresult
    return render_template('predict.html',prediction=result)

@app.route("/about")
def about():
    return render_template('about.html')
@app.route("/result",methods=['GET', 'POST'])
def result():
    
    return render_template('result.html',bedrooms=bedrooms)
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