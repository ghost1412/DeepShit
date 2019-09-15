# imports
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm, trange

class LinearRegressionUsingGD:

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations

    def scaleData(self, x):
        sc = StandardScaler()
        X = sc.fit_transform(x)
        return X

    def fit(self, x, y, Ltype=None, Lambda=1000):
 
        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= np.dot((self.eta / m), gradient_vector)
            cost = np.sum((residuals ** 2)) / (2 * m)
            #regularization l2
            if Ltype == 'L2':
            	cost += (1/(2*len(y))*np.sum((self.w_**2)))*Lambda
            if Ltype == 'L1':
                cost += (1/(2*len(y))*np.sum((self.w_)))*Lambda
            self.cost_.append(cost)
        return self.cost_

    def predict(self, x):
        return np.dot(x, self.w_)
