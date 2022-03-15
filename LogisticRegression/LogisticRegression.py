
from Model import Model
import numpy as np
import Utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogisticRegression(Model):

    def __init__(self, lr=0.01, th=0.001, max_iter=10000):
        super(LogisticRegression, self).__init__()
        self.W = None
        self.lr = lr
        self.th = th
        self.max_iter = max_iter

    def grad(self, X, Y):
        p = Y-Utils.sig(np.dot(X,self.W.T))
        return np.dot(p.T,X)

    def fit(self, X, Y):
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        Y = Y.reshape((-1,1))
        self.W = np.random.rand(1,X.shape[1])
        for i in range(self.max_iter):
            j = self.grad(X, Y)
            if abs(j.sum()) < self.th:
                break
            self.W = self.W + self.lr * j

    def predict(self, X):
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        y_ = np.dot(X, self.W.T)
        return np.array([1 if a >= 0 else 0 for a in y_])

