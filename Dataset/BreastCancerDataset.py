__author__ = 'Ashutosh Kshirsagar'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from  importlib.resources import path
from Dataset.Datasets import Dataset

class BreastCancer(Dataset):
    """
    Breast Cancer Wisconsin (Diagnostic) Data Set
    @misc{Dua:2019 ,
    author = "Dua, Dheeru and Graff, Casey",
    year = "2017",
    title = "{UCI} Machine Learning Repository",
    url = "http://archive.ics.uci.edu/ml",
    institution = "University of California, Irvine, School of Information and Computer Sciences" }

    Download link: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
    """

    def __init__(self, path='../Files/BreastCancerDS.csv'):
        super(BreastCancer,self).__init__()
        self.path = path

    def get_dataset(self):
        df = pd.read_csv(self.path)
        X_ = df[['radius_mean', 'texture_mean', 'perimeter_mean',
                 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                 'perimeter_worst', 'area_worst', 'smoothness_worst',
                 'compactness_worst', 'concavity_worst', 'concave points_worst',
                 'symmetry_worst', 'fractal_dimension_worst']].to_numpy()
        Y_ = df['diagnosis'].map({'M': 1, 'B': 0}).to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    def create_kfold(self, k=10):
        df = pd.read_csv(self.path)
        X = df[['radius_mean', 'texture_mean', 'perimeter_mean',
                 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                 'fractal_dimension_se', 'radius_worst', 'texture_worst',
                 'perimeter_worst', 'area_worst', 'smoothness_worst',
                 'compactness_worst', 'concavity_worst', 'concave points_worst',
                 'symmetry_worst', 'fractal_dimension_worst']].to_numpy()
        Y = df['diagnosis'].map({'M': 1, 'B': 0}).to_numpy()
        return self.k_fold(X, Y,k)
