# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:35:05 2022

@author: aceso
"""

import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer # need this module to use below fx
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
import pickle
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error


# Constant
TRAIN = os.path.join(os.getcwd(), "Data", "cases_malaysia_train.csv")
TEST = os.path.join(os.getcwd(), "Data", "cases_malaysia_test.csv")
SCALER = os.path.join(os.getcwd(), "Saved", "minmax.pkl")

#%% Classes

class EDA():
    
    def data_clean(self,data):
        data["cases_new"] = pd.to_numeric(data["cases_new"], errors="coerce")
        chain = IterativeImputer()
        data["cases_new"] = chain.fit_transform(data)
        
        return data
    
    def data_scaling(self,data, title):
        minmax = MinMaxScaler() # minmax because in this data there's no -ve value
        data = minmax.fit_transform(np.expand_dims(data, -1))
        sns.distplot(data)
        plt.title(title)
        plt.legend()
        plt.show()
        pickle.dump(minmax, open(SCALER, "wb"))
        return data
    
    def data_split(self, train, test, window_size):
        # Training Data
        X_train = []
        y_train = []
        for i in range(window_size, len(train)): #(window_size, max number or rows)
            X_train.append(train[i-window_size:i, 0])
            y_train.append(train[i,0])
            
        X_train = np.array(X_train)
        X_train = np.expand_dims(X_train, -1) # dimension expansion 
        y_train = np.array(y_train)
            
        # Testing data
        temp = np.concatenate((train, test)) # using np.concat since both in array
        temp = temp[-(window_size + len(test)):] # last 60(training) + 96(test) == -156
        
        X_test = []
        y_test = []
        for i in range(window_size, len(temp)):
            X_test.append(temp[i-window_size:i,0])
            y_test.append(temp[i,0])
            
        X_test = np.array(X_test)
        X_test = np.expand_dims(X_test, -1) # dimension expansion
        y_test = np.array(y_test)
            
        return X_train, y_train, X_test, y_test
        
class ModelConfig():
    
    def lstm(self, nodes, X_train):
        model = Sequential()
        model.add(LSTM(nodes, return_sequences=(True), input_shape=(X_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(nodes, return_sequences=(True)))
        model.add(Dropout(0.2))
        model.add(LSTM(nodes, return_sequences=(True)))
        model.add(Dropout(0.2))
        model.add(LSTM(nodes))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation = "relu")) # since there's no -ve value use relu
        model.summary()
        
        return model
    
class Performance():
        
    def mape(self, y_true, y_pred):
        print(f"MAPE prediction is: {(mean_absolute_error(y_true, y_pred)/sum(abs(y_true))) * 100}%")
        
















