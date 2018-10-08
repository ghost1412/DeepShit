# IMPORTING IMPORTANT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from preprocessing import data_process
import pandas_datareader.data as web
from datetime import datetime
from model import *
from sklearn.preprocessing import MinMaxScaler

date_range=7

def trainData():

	msft = web.DataReader('MSFT', 'yahoo', start=datetime(2016, 1, 1), end=datetime(2017, 12, 31))
	msft['HighMinusLow'] = pd.DataFrame(msft['High']).sub(msft['Low'], axis=0)
	sc=MinMaxScaler()
	msft['VolumeScaled'] = sc.fit_transform(pd.DataFrame(msft['Volume']))
	training_set1 = pd.DataFrame(msft[['Open', 'HighMinusLow', 'Close', 'VolumeScaled']]).mean(axis=1)
	training_set = np.reshape(training_set1.values, (len(training_set1), 1))
	training_set.shape

	return training_set

def valiData():

	msft = web.DataReader('MSFT', 'yahoo', datetime(2018, 1, 1), datetime(2018, 8, 30))
	validate_set=pd.DataFrame(msft['Close'].values)
	validate_set.shape
	
	return validate_set

def normalizeData(training_set, validate_set):

	sc=MinMaxScaler()
	training_scaled=sc.fit_transform(training_set)
	validate_scaled=sc.fit_transform(validate_set)

	return training_scaled, validate_scaled, sc


def plotData(y_val_unscaled, prediction):
	#%matplotlib inline
	import matplotlib.pyplot as plt

	plt.plot(y_val_unscaled, color='blue', label='actual stock price')
	plt.plot(prediction, color='green', label='predicted stock price')

	plt.title('Prediction of Microsoft stock price')
	plt.ylabel('Price (Jan-Aug,2018)')
	plt.xlabel('Days')
	plt.legend(['actual price', 'predicted price'], loc='upper left')
	plt.show()	


def init():

	training_set = trainData()
	validate_set = valiData()
	training_scaled, validate_scaled, sc = normalizeData(training_set, validate_set)
	X_train, target=data_process(training_scaled, date_range)
	X_train_input=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1) )
	X_train_input.shape

	X_val, y_val=data_process(validate_scaled, date_range)
	X_val_input=np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1) )
	X_val_input.shape
	
	reg_model = model(date_range)
	reg_model.compile(optimizer='adam', loss='mean_squared_error')
	reg_model.fit(X_train_input, target, validation_data=(X_val_input, y_val), batch_size=1, nb_epoch=100, verbose=1)
	prediction_scaled=reg_model.predict(X_val_input)
	prediction=sc.inverse_transform(prediction_scaled)
	y_val_unscaled=sc.inverse_transform(y_val.reshape(-1,1))
	plotData(y_val_unscaled, prediction)

if __name__ == "__main__":
	init()





























































































