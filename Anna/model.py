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
df=pd.read_csv("project_files/analytical_table.csv")

lb_encoder = LabelEncoder()
df['category'] = lb_encoder.fit_transform(df['category'])
df['type'] = lb_encoder.fit_transform(df['type'])
df['sub_type'] = lb_encoder.fit_transform(df['sub_type'])
df['state'] = lb_encoder.fit_transform(df['state'])
df['locality'] = lb_encoder.fit_transform(df['locality'])
# Split the labels and features in original dataset
features = df.drop("price", axis=1)
labels = df["price"].copy()
for column in features:
    if features[column].dtypes == 'int64':
        features[column] = StandardScaler().fit_transform(features[column].values.reshape(-1, 1))
for column in features:
    if features[column].dtypes == 'float64':
        features[column] = StandardScaler().fit_transform(features[column].values.reshape(-1, 1))
# Call train_test_split function from sklearn library to split the dataset randomly
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=1/3, random_state=126)
class ANNRegressor(BaseEstimator, RegressorMixin):
    # Constructor to instantiate default or user-defined values
    def __init__(self, in_features=16, num_hidden=1, num_neurons=36, epochs=75, 
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

X_train = np.asarray(X_train).astype(np.float32)
Y_train = np.asarray(Y_train).astype(np.float32)
annRegressor = ANNRegressor(in_features=X_train.shape[1], num_hidden=25, num_neurons=48, epochs=35, verbose=1)
annRegressor.fit(X_train, Y_train)
import pickle
pickle.dump(annRegressor, open('models/annmodel.pkl', 'wb'))
 
loaded_model = pickle.load(open("models/annmodel.pkl", "rb"))

# [[28435124.]]
# Method to display model evaluation metrics
def display_model_metrics(label, predictions):
    # The mean absolute error
    print("Mean absolute error: %.4f\n" % mean_absolute_error(label, predictions))

    # The mean squared error
    print("Root mean squared error: %.4f\n" % np.sqrt(mean_squared_error(label, predictions)))

    # The coefficient of determination: 1 is perfect prediction R^2
    print("Coefficient of determination: %.4f\n" % r2_score(label, predictions))

predictions = annRegressor.predict(X_test)

display_model_metrics(Y_test, predictions[:,-1])