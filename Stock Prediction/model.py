from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM

def model(dateRange):
	
	reg_model=Sequential()
	reg_model.add(LSTM(128, input_shape=(dateRange, 1), return_sequences=True)) # each data point consists of 7 days of closing price data.
	reg_model.add(LSTM(32, return_sequences=True))
	reg_model.add(LSTM(16))
	reg_model.add(Dense(1, activation='linear'))
	
	return reg_model

