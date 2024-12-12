import pandas as pd 
import numpy as np 
import xgboost as xgb 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import ffnn_utils as fut
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
from sklearn.ensemble import IsolationForest
from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF



'''
Custom Function to perform all of the metric analysis at once for supervised learning
@param - ytrue: The true labels of the dataset we want to evaluate for
@param - y_pred: The redicted labels from the machine learning model
@param - y_prob: The probalbility that a prediction falls into that class
@return - meterics: a dictionary of the metrics computed for a model
'''
def evaluate_model_performance(y_true, y_pred, y_prob):
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division= 1)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=1)
    if len(set(y_true)) < 2:
        auc_roc = -1
    else:     
        auc_roc = roc_auc_score(y_true, y_prob)
    # Return metrics in a dictionary
    metrics = {
        'Accuracy': round(accuracy, 4),
        'Precision': round(float(precision), 4),
        'Recall': round(float(recall), 4),
        'F1 Score': round(float(f1), 4),
        'AUC-ROC': round(float(auc_roc), 4)
    }
    return metrics


'''
A function to calculate the average of the metric results for multiple runs of a model
@param - meterics_list: A list of the metrics that have been reported for multiple iterations of a model 
@return - avg_meterics: A single dictionary of meterics that have been averaged across the number of iterations 
'''
def average_metrics(metrics_list):

    avg_metrics = {key: 0 for key in metrics_list[0]}
    for metrics in metrics_list:
        for key in avg_metrics:
            avg_metrics[key] += metrics[key]
    num_iterations = len(metrics_list)
    for key in avg_metrics:
        avg_metrics[key] /= num_iterations
    
    return avg_metrics

#### XGBOOST METHODS

'''
A function to build an XGBOost model with default parameters
@param - features: The feature inputs to the model 
@param - target: The target inputs to the model
@return xgbModel: The fitted model with full weights and parameters
'''
def build_xgbModel(features, target): 
    xgbModel = xgb.XGBClassifier()
    xgbModel.fit(features, target)
    return xgbModel

'''
A function used in the pipelining of the XGBoost Model 
@param - X_train: The feature data that should be used during model training
@param - X_test: The target data that should be used during model training
@param - y_train: The feature data that should be used during model testing
@param - y_test: The feature data that should be used during model testing
@return - meterics: A dictionary of metrics that has the accuracy, recall, percision, f1, and AUC-ROC scores of the model
'''
def xgbFull(X_train, X_test, y_train, y_test):
    model = build_xgbModel(X_train, y_train)

    preds = model.predict(X_test)

    pred_proba = model.predict_proba(X_test)[:,1]

    metrics = evaluate_model_performance(y_test, preds, pred_proba)
    return metrics


### SVM Model 


'''
Function that creates a Support Vector Machine with kernel support but otherwise default parameters 
@param - features: The feature inputs to the model 
@param - target: The target inputs to the model 
@return - svm_model: A fitted SVM model with full weights and parameters
'''
def build_svm(features, target, kernel): 
    svm_model = SVC(kernel=kernel, probability=True)
    svm_model.fit(features, target)
    return svm_model



'''
Function used in pipelining of the SVM model 
@param - X_train: The feature inputs that should be used during model training
@param - X_test: The target inputs that should be used during model training 
@param - y_train: The feature inputs that should be used during model testing 
@param - y_test: The target inputs that should be used during model testing 
@param - kernel: The desired kernel that the SVM model should use
@return - meterics: A dictionary of metrics that has the accuracy, recall, percision, f1, and AUC-ROC scores of the model
'''
def svmFull(X_train, X_test, y_train, y_test, kernel):
    model = build_svm(X_train, y_train, kernel)

    preds = model.predict(X_test)

    pred_proba = model.predict_proba(X_test)[:,1]

    metrics = evaluate_model_performance(y_test, preds, pred_proba)

    return metrics

### UNWORKING FFNN 

def ffnn(features, targets): 
    fut.run_ffnn_model(features,targets, 2)

### Logistic Regression 
    
'''
Function that creates a Logestic Regression Classifier with default parameters 
@param - features: The feature inputs to the model 
@param - target: The target inputs to the model 
@return - logreg_model: A fitted Logistic Regression model with full weights and parameters
'''
def build_LogReg(features, target): 
    logreg_model = LogisticRegression(max_iter = 1000000)
    logreg_model.fit(features, target)
    return logreg_model

'''
Function used in pipelining of the Logitic Regression model 
@param - X_train: The feature inputs that should be used during model training
@param - X_test: The target inputs that should be used during model training 
@param - y_train: The feature inputs that should be used during model testing 
@param - y_test: The target inputs that should be used during model testing 
@return - meterics: A dictionary of metrics that has the accuracy, recall, percision, f1, and AUC-ROC scores of the model
'''
def LogRegFull(X_train, X_test, y_train, y_test):
    model = build_LogReg(X_train, y_train)
    preds = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:,1]
    metrics = evaluate_model_performance(y_test, preds, pred_proba)
    return metrics

'''
Function that is used for visualizaion of the confusion matrix for a models predictions 
@param - preds: The models predicted values 
@param - ture: The true labels fo the dataset
'''
def build_cm(preds, true): 
    con = confusion_matrix(y_pred=preds, y_true=true)
    disp = ConfusionMatrixDisplay(confusion_matrix=con)
    disp.plot()
    plt.show()


##########################################
#  Outlier Detection/Anomolay Detection  # 
#  Since it is unsupervised, there are   #
#  no scoring functions                  #
##########################################


'''
Function that builds and fits an Isolation Forest model with default parameters
@param - data: The dataset that is being used for anomaly detection 
@return - data_copy: The original dataset after anomaly detection with an additional column 
to denote whether or not the column is anomolus
'''
def build_isolation_forest(data): 
    data_copy = data.copy()
    #Declare the model
    isoModel = IsolationForest()
    #fit the model
    isoModel.fit(data)
    #Store relevant informtaion in a copy of the data so that we can view it 
    data_copy['anomaly'] = isoModel.predict(data)
    #return the copy
    return data_copy

'''
Function that builds and fits an \Angle Based Outlier Detection model with default parameters
@param - data: The dataset that is being used for anomaly detection 
@return - data_copy: The original dataset after anomaly detection with an additional column 
to denote whether or not the column is anomolus
'''
def build_abod(data): 
    data_copy = data.copy() 
    #Declare the model 
    abdoModel = ABOD()
    #fit the model 
    abdoModel.fit(data)
    #store the relevant information in a copy of the data so that we can view it 
    data_copy['label'] = abdoModel.labels_
    # data_copy['threshold'] = abdoModel.threshold_
    return data_copy


'''
Function that builds and fits an Cluster Based Local Outlier Factor model with default parameters
@param - data: The dataset that is being used for anomaly detection 
@return - data_copy: The original dataset after anomaly detection with an additional column 
to denote whether or not the column is anomolus
'''
def build_cblof(data): 
    data_copy = data.copy()
    cblofModel = CBLOF() 
    cblofModel.fit(data)
    data_copy['label'] = cblofModel.labels_
    return data_copy
