
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from  importlib.resources import path

class Dataset:

    def __init__(self):
        pass

    def get_dataset(self):
        pass

    def k_fold(self,X, Y, k):
        j = int(X.shape[0] / k)
        DS = []
        for i in range(k):
            test = (X[i * j:(i + 1) * j], Y[i * j:(i + 1) * j])
            if i * j == 0:
                train = (X[(i + 1) * j:], Y[(i + 1) * j:])
            elif (i + 1) * j - 1 >= X.shape[0]:
                train = (X[:i * j], Y[:i * j])
            else:
                train = (np.append(X[:i * j], X[(i + 1) * j:], axis=0), np.append(Y[:i * j], Y[(i + 1) * j:], axis=0))
            DS.append((train, test))
        return DS




