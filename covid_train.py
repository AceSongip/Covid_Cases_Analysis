# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:35:48 2022

@author: aceso
"""

import pandas as pd
import numpy as np 
import os
import pickle
import datetime
import matplotlib.pyplot as plt
from covid_classes import EDA, ModelConfig, Performance
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# Constant
TRAIN = os.path.join(os.getcwd(), "Data", "cases_malaysia_train.csv")
TEST = os.path.join(os.getcwd(), "Data", "cases_malaysia_test.csv")
LOG = os.path.join(os.getcwd(), "Log")
log_dir = os.path.join(LOG, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
MODEL = os.path.join(os.getcwd(), "Saved", "model.h5")
SCALER = os.path.join(os.getcwd(), "Saved", "minmax.pkl")

#%% EDA
# Import data
df_train = pd.read_csv(TRAIN) # no need parse = date because data alreadin in sequence
df_test = pd.read_csv(TEST)

# Inspect data
# Train data
df_train = df_train.drop(columns="date")
df_train["cases_new"].info() # dtype non-numerical
df_train["cases_new"].describe() # There are 680 data

# Test data
df_test = df_test.drop(columns="date")

# visualize the train data
plt.figure()
plt.plot(df_train["cases_new"])
plt.show()

# data cleaning
eda = EDA()
# Train data
new_train = df_train.copy()
train = eda.data_clean(new_train)
train = train.loc[:,"cases_new"]
# Test data
new_test = df_test.copy()
test = eda.data_clean(new_test)
test = test.loc[:,"cases_new"]

# data scalling
scaled_train = eda.data_scaling(train, title="Minmax Training") # scaled_training
scaled_test = eda.data_scaling(test, title="Minmax Testing") # scaled_training

X_train, y_train, X_test, y_test = eda.data_split(scaled_train, scaled_test, window_size=30)

#%% Model building
nn = ModelConfig()
model = nn.lstm(nodes = 64, X_train=X_train)
model.compile(optimizer="adam", loss="mse", metrics="mse")
plot_model(model)

# EarlyStopping
es = EarlyStopping(monitor="loss", patience=5)
# Tensorboard
tensorboard = TensorBoard(log_dir = log_dir)

hist = model.fit(X_train, y_train, epochs=100, batch_size=100,
                 callbacks=[tensorboard, es])

# Model saving
model.save(MODEL)

#%% Model Evaluation and Analysis

predicted = []
for i in X_test: 
    predicted.append(model.predict(np.expand_dims(i, axis=0))) 
                                                               
predicted = np.array(predicted)

y_pred = np.squeeze(predicted, axis=1)
y_true = np.expand_dims(y_test, axis=-1)

evaluate = Performance()
evaluate.mape(y_true=y_true, y_pred=y_pred)

#%% Visualize y_pred vs y_true

plt.figure()
plt.plot(y_pred) # reshape predicted to (96,1)
plt.plot(y_true)
plt.legend(["predicted", "actual"])
plt.title("Predicted vs Actual")
plt.show()

# After inverse
with open(SCALER, "rb") as f:
    mms = pickle.load(f)

y_true_inverse = mms.inverse_transform(y_true)
y_pred_inverse = mms.inverse_transform(y_pred)

plt.figure()
plt.plot(y_pred_inverse) # reshape predicted to (96,1)
plt.plot(y_true_inverse)
plt.legend(["predicted", "actual"])
plt.title("Predicted vs Actual")
plt.show()



