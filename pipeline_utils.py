import model_utils as mods 
import posion_utils as pu 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import preprocessing_utils as put


def test_posion(posionType, dataset): 
    X_train, X_test, y_train, y_test = put.heart()
    xgboost_list = []
    svm_list = []
    logreg_list = []
    num_iter = 100

    for i in range(num_iter): 
        xgboost_list.append(mods.xgbFull(X_train, X_test, y_train, y_test))
        svm_list.append(mods.svmFull(X_train, X_test, y_train, y_test, 'poly'))
        logreg_list.append(mods.LogRegFull(X_train, X_test, y_train, y_test))

    # Now average the metrics for each model
    xgboost_avg_metrics = mods.average_metrics(xgboost_list)
    svm_avg_metrics = mods.average_metrics(svm_list)
    logreg_avg_metrics = mods.average_metrics(logreg_list)

    # Print the averaged results for each model
    print("Averaged XGBoost Metrics:", xgboost_avg_metrics)
    print("Averaged SVM Metrics:", svm_avg_metrics)
    print("Averaged Logistic Regression Metrics:", logreg_avg_metrics)
        