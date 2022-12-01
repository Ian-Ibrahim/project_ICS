# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import sklearn

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
#tensorflow importing
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import mean_squared_logarithmic_error, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
#import dataset
df=pd.read_csv("project_files/cleaned_listings.csv")
df['listdate']= pd.to_datetime(df['listdate'])
df["list_year"]=df['listdate'].dt.year
df["list_month"]=df['listdate'].dt.month
df.drop('listdate', inplace=True, axis=1)
print (df.state.unique())
lb_encoder = LabelEncoder()
df['category'] = lb_encoder.fit_transform(df['category'])
df['type'] = lb_encoder.fit_transform(df['type'])
df['sub_type'] = lb_encoder.fit_transform(df['sub_type'])
df['state'] = lb_encoder.fit_transform(df['state'])
df['locality'] = lb_encoder.fit_transform(df['locality'])
