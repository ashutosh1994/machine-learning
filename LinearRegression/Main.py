__author__ = 'Ashutosh Kshirsagar'
from LinearRegression import LinearRegressionCF, LinearRegressionGD
from Dataset.BikeSharingDataset import BikeSharing
from Metrics import RRSE

if __name__ == '__main__':

    X_train, X_test, y_train, y_test = BikeSharing().get_dataset()
    lr = LinearRegressionCF()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    print(f"RMSE {RRSE(y_test,y_pred)}")

    X_train, X_test, y_train, y_test = BikeSharing().get_dataset()
    lr = LinearRegressionGD(lr=0.0001)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(f"RMSE {RRSE(y_test, y_pred)}")