from flask import Flask, request, render_template
import numpy as np
import re
import sys 
import os
import base64
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import json
import os
import sqlite3
from flask import Flask, redirect, request, url_for
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)

from oauthlib.oauth2 import WebApplicationClient
import requests

from db import init_db_command
from user import User

app = Flask(__name__)
app.secret_key = "supper secret key ics proj"
app.app_context().push()
# Configuration
GOOGLE_CLIENT_ID = "357509455628-l0gl7fi8gj874piclf071scd90j07kjs.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
GOOGLE_DISCOVERY_URL = (
    "https://accounts.google.com/.well-known/openid-configuration"
)

login_manager = LoginManager()
login_manager.init_app(app)
# Naive database setup
try:
    init_db_command()
except sqlite3.OperationalError:
    # Assume it's already been created
    pass

# OAuth 2 client setup
client = WebApplicationClient(GOOGLE_CLIENT_ID)
@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

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


loaded_model = pickle.load(open("models/annmodel.pkl", "rb"))
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,-1)
    result = loaded_model.predict(to_predict)
    print(result)
    return result[0]
 
@app.route("/")
def home():
    if current_user.is_authenticated:
        return (
            # "<p>Hello, {}! You're logged in! Email: {}</p>"
            # "<div><p>Google Profile Picture:</p>"
            # '<img src="{}" alt="Google profile pic"></img></div>'
            # '<a class="button" href="/logout">Logout</a>'.format(
            #     current_user.name, current_user.email, current_user.profile_pic
            # )
            render_template('index.html')
        )
    else:
        return '<a class="button" href="/login">Google Login</a>'
    

@app.route("/predict",methods=['GET','POST'])
@login_required
def predict():
    rentalResult=0
    saleResult=0
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
        category=0
        to_predict_list.append(category)
        houseType=request.form.get("type")
        to_predict_list.append(houseType)
        houseSubType=request.form.get("sub_type")
        to_predict_list.append(houseSubType)
        state=request.form.get("county")
        to_predict_list.append(state)
        locality=request.form.get("locality")
        to_predict_list.append(locality)
        pre_covid=0;
        to_predict_list.append(pre_covid)
        list_year=request.form.get("year")
        to_predict_list.append(list_year)
        list_month=request.form.get("month")
        to_predict_list.append(list_month)
        to_predict_list = list(map(int, to_predict_list))
        rentalResult = ValuePredictor(to_predict_list) 
        rentalResult="{:f}".format(rentalResult[0])
        to_predict_list[8]=1
        saleResult=ValuePredictor(to_predict_list)
        saleResult="{:f}".format(saleResult[0])
    return render_template('predict.html',salePrediction=saleResult,rentPrediction=rentalResult)

@app.route("/about")
def about():
    return render_template('about.html')
@app.route("/profile")
@login_required
def profile():
    return render_template('profile.html')
@app.route("/result",methods=['GET', 'POST'])
def result():
    return render_template('result.html',bedrooms=bedrooms)
@app.route("/login")
def login():
    # Find out what URL to hit for Google login
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    # Use library to construct the request for Google login and provide
    # scopes that let you retrieve user's profile from Google
    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=request.base_url + "/callback",
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)
    # return render_template('authentication/login.html')
@app.route("/login/callback")
def callback():
    # Get authorization code Google sent back to you
    code = request.args.get("code")

    # Find out what URL to hit to get tokens that allow you to ask for
    # things on behalf of a user
    google_provider_cfg = get_google_provider_cfg()
    token_endpoint = google_provider_cfg["token_endpoint"]

    # Prepare and send request to get tokens! Yay tokens!
    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code,
    )
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )

    # Parse the tokens!
    client.parse_request_body_response(json.dumps(token_response.json()))

    # Now that we have tokens (yay) let's find and hit URL
    # from Google that gives you user's profile information,
    # including their Google Profile Image and Email
    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)

    # We want to make sure their email is verified.
    # The user authenticated with Google, authorized our
    # app, and now we've verified their email through Google!
    if userinfo_response.json().get("email_verified"):
        unique_id = userinfo_response.json()["sub"]
        users_email = userinfo_response.json()["email"]
        picture = userinfo_response.json()["picture"]
        users_name = userinfo_response.json()["given_name"]
    else:
        return "User email not available or not verified by Google.", 400

    # Create a user in our db with the information provided
    # by Google
    user = User(
        id_=unique_id, name=users_name, email=users_email, profile_pic=picture
    )

    # Doesn't exist? Add to database
    if not User.get(unique_id):
        User.create(unique_id, users_name, users_email, picture)

    # Begin user session by logging the user in
    login_user(user)

    # Send user back to homepage
    return redirect(url_for("home"))
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

@app.route("/register")
def register():
    return render_template('authentication/register.html')

def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()
# handle 404 error
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404
if __name__ == "__main__":
    app.run(debug=True,ssl_context="adhoc")