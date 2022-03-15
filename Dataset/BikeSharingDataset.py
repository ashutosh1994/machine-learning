__author__ = 'Ashutosh Kshirsagar'


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from  importlib.resources import path
from Dataset.Datasets import Dataset

class BikeSharing(Dataset):
    """

    Bike Sharing Dataset Data Set
    @article{
    year={2013},
    issn={2192-6352},
    journal={Progress in Artificial Intelligence},
    doi={10.1007/s13748-013-0040-3},
    title={Event labeling combining ensemble detectors and background knowledge},
    url={[Web Link]},
    publisher={Springer Berlin Heidelberg},
    keywords={Event labeling; Event detection; Ensemble learning; Background knowledge},
    author={Fanaee-T, Hadi and Gama, Joao},
    pages={1-15}
    }

    Download link: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
    """

    def __init__(self, path='../Files/BikeSharingDailyDS.csv',Daily=True):
        super(BikeSharing,self).__init__()
        self.path = path

    def get_dataset(self):
        df = pd.read_csv(self.path)
        X_ = df[['season', 'yr', 'mnth', 'holiday', 'weekday',
               'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
               'casual', 'registered', 'cnt']].to_numpy()
        Y_ = df['cnt'].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    def create_kfold(self, k=10):
        df = pd.read_csv(self.path)
        X_ = df[['season', 'yr', 'mnth', 'holiday', 'weekday',
                 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
                 'casual', 'registered', 'cnt']].to_numpy()
        Y_ = df['cnt'].to_numpy()
        return self.k_fold(X_, Y_,k)
