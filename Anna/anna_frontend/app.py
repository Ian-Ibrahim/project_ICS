import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import Pickle
app = Flask(__name__)
model = pickle.load(open('../models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/')
def hello():
    return 'Hello, World!'