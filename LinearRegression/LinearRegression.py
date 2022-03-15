__author__ = 'Ashutosh Kshirsagar'

from Model import Model
import numpy as np

class LinearRegression(Model):

    def __init__(self):
        super(LinearRegression,self).__init__()

    def fit(self, X,y):
        pass

    def predict(self, X):
        pass

class LinearRegressionCF(LinearRegression):
    """
    Closed form implementation of Linear Regression
    """

    def __init__(self):
        super(LinearRegressionCF, self).__init__()
        self.theta = None

    def fit(self, X, Y):
        self.theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))  # closed form
        return self.theta

    def predict(self, X):
        return np.dot(X, self.theta)

class LinearRegressionGD(LinearRegression):
    """
    Closed form implementation of Linear Regression
    """

    def __init__(self, lr=0.001, th=0.1, max_iter=10000):
        super(LinearRegressionGD, self).__init__()
        self.theta = None
        self.lr = lr
        self.th = th
        self.max_iter = max_iter

    def grad(self,X,y):
        return np.dot(np.dot(X.T, X), self.theta) - np.dot(X.T, y)

    def fit(self, X, Y):
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        Y = Y.reshape((-1, 1))
        self.theta = np.random.rand( X.shape[1],1)
        for i in range(self.max_iter):
            j = self.grad(X, Y)
            if abs(j.sum()) < self.th:
                break
            self.theta = self.theta + self.lr * j
        j = 0

    def predict(self, X):
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        return np.dot(X, self.theta)