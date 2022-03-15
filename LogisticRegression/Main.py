from Dataset.BreastCancerDataset import BreastCancer
from sklearn.metrics import accuracy_score
from LogisticRegression import LogisticRegression

if __name__ == '__main__':

    X_train, X_test, y_train, y_test = BreastCancer().get_dataset()
    lr = LogisticRegression(lr=0.03)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    print(f"Accuracy {accuracy_score(y_test,y_pred)}")