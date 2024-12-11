import model_utils as mods 
import poison_utils as pu 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import preprocessing_utils as put
import majority_voting as mv


def test_posion_1(posionType, percent, number, dataset, mode, style): 
    if dataset == 'heart':
        X_train, X_test, y_train, y_test = put.heart_poison(posionType, percent, number, mode,style)
    if dataset == 'loan': 
        X_train, X_test, y_train, y_test = put.loan_poison(posionType, percent, number, mode, style)
    if dataset == 'cancer': 
        X_train, X_test, y_train, y_test = put.cancer_poison(posionType, percent, number, mode,style)
    if dataset == 'machine': 
        X_train, X_test, y_train, y_test = put.machine_poison(posionType, percent, number, mode,style)
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



'''
Function to run all three poison types in a dataset and perform the anomaly detection and output the 
points that have been detected as anomalous
'''
def test_poison_demo(data, poison_type, dataset): 
    if poison_type == 'FLIP': 
        data2 = pu.flip_random_labels(data, 0.15, dataset)
        data2 = mv.detect_poison(data2)
        return data2['Poisoned'].value_counts()
    if poison_type == "INJECT": 
        data2 = pu.inject_new(data, 15, 'DISTRIBUTION')
        data2 = mv.detect_poison(data2)
        return data2['Poisoned'].value_counts()
    if poison_type == 'TAMPER': 
        data2 = pu.tamper_rows(data, .1, 'DISTRIBUTION')
        data2 = mv.detect_poison(data2)
        return data2['Poisoned'].value_counts()
    return None
