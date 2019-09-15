import csv
import pandas as pd
import numpy as np
from preprocessor import preprocessor
from regression import LinearRegressionUsingGD
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from sklearn.linear_model import LinearRegression
#data = preprocessor.processData()
with open('ozone.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
with open('so2.csv', 'r') as f:
    reader = csv.reader(f)
    data1 = list(reader)
with open('pm2.5.csv', 'r') as f:
    reader = csv.reader(f)
    data2 = list(reader)
x = np.array(data)[1:, 3:].astype(np.float)
Y = np.array(data)[1:, 2].astype(np.float)
reg = LinearRegressionUsingGD()
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
Y = min_max_scaler.fit_transform(np.reshape(Y, (len(Y), 1)))
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state=0)
cost = reg.fit(X_train, np.reshape(y_train, (len(y_train), 1)))
itera = list(range(0, 1000))
y_pred = reg.predict(X_test)
mse = (np.square(np.reshape(y_test, (len(y_test), 1))-np.reshape(y_pred, (len(y_pred), 1)))).mean()/2

x = np.array(data1)[1:, 3:].astype(np.float)
Y = np.array(data1)[1:, 2].astype(np.float)
reg = LinearRegressionUsingGD()
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
Y = min_max_scaler.fit_transform(np.reshape(Y, (len(Y), 1)))
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state=0)
cost1 = reg.fit(X_train, np.reshape(y_train, (len(y_train), 1)))
itera = list(range(0, 1000))
y_pred = reg.predict(X_test)
mse = (np.square(np.reshape(y_test, (len(y_test), 1))-np.reshape(y_pred, (len(y_pred), 1)))).mean()/2


x = np.array(data2)[1:, 3:].astype(np.float)
Y = np.array(data2)[1:, 2].astype(np.float)
reg = LinearRegressionUsingGD()
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
Y = min_max_scaler.fit_transform(np.reshape(Y, (len(Y), 1)))
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state=0)

cost2 = reg.fit(X_train, np.reshape(y_train, (len(y_train), 1)))
itera = list(range(0, 1000))
y_pred = reg.predict(X_test)
mse = (np.square(np.reshape(y_test, (len(y_test), 1))-np.reshape(y_pred, (len(y_pred), 1)))).mean()/2

ct = [cost, cost1, cost2]
gasType = ['Ozone', 'SO2', 'PM2.5']
for i in range(3):
    plt.plot(itera, ct[i],label = 'Gas %s'%gasType[i])
plt.legend()
plt.show()
'''df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()'''


