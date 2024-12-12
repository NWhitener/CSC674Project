import model_utils as mods 
import poison_utils as pu 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import preprocessing_utils as put
import majority_voting as mv



'''
A pipelining function that tests a specific type of poision on a dataset 
@param - poisonType: The type of poison that should be used. Can take values of FLIP, TAMPER, INJECT, MISDIRECTION
@param - percent: The percent of data that should be poisoned, used for the TAMPER, and FLIP poison types 
@param - number: The number of rows to poision, used for the MISDERICTION and INJECT poison types 
@param - dataset: The name of the dataset that the poison benchmarking should occur on 
@param - mode: The mode of TAMPER, and INJECT that should occur. Can take values of RANDOM, DISTRIBUTION, or MALICOUS 
@param - style: The style of the dataset poisoning. Used to differentiate between returning the X_train style or the full dataset 
'''
def test_posion_1(posionType, percent, number, dataset, mode, style): 
    if dataset == 'HEART':
        X_train, X_test, y_train, y_test = put.heart_poison(posionType, percent, number, mode,style)
    if dataset == 'LOAN': 
        X_train, X_test, y_train, y_test = put.loan_poison(posionType, percent, number, mode, style)
    if dataset == 'CANCER': 
        X_train, X_test, y_train, y_test = put.cancer_poison(posionType, percent, number, mode,style)
    if dataset == 'MACHINE': 
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
Function to test the anomaly/outlier detection methods to be used in the in-class demo 
@param - data: The dataset that should be used for testing 
@param - poison_type: The type of poison that should be used 
@param - dataset: The dataset name to handle special cases 
'''
def test_poison_demo(data, poison_type, dataset): 
    if poison_type == 'FLIP': 
        data2 = pu.flip_random_labels(data, 0.15, dataset)
        print(data2['Tampered'].value_counts())
        data2 = mv.detect_poison(data2)
        return data2[data2['Tampered'] == 1]['Poisoned'].value_counts()
    if poison_type == "INJECT": 
        data2 = pu.inject_new(data, 20, 'DISTRIBUTION')
        data2 = mv.detect_poison(data2)
        return data2[data2['Tampered'] == 1]['Poisoned'].value_counts()
    if poison_type == 'TAMPER': 
        data2 = pu.tamper_rows(data, .1, 'DISTRIBUTION')
        data2 = mv.detect_poison(data2)
        return data2[data2['Tampered'] == 1]['Poisoned'].value_counts()
    if poison_type == 'MISDIRECTION': 
        data2 = pu.misdirection(data, 20, 'MAJOR')
        data2 = mv.detect_poison(data2)
        return data2[data2['Tampered'] == 1]['Poisoned'].value_counts()
