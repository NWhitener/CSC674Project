import pandas as pd 
import numpy as np 
import xgboost as xgb 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import ffnn_utils as fut


def evaluate_model_performance(y_true, y_pred, y_prob):
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
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


#### XGBOOST METHODS

def build_xgbModel(features, target): 
    xgbModel = xgb.XGBClassifier()
    xgbModel.fit(features, target)
    return xgbModel


def score_xgbModel(model, X_test, y_test):
    return model.score(X_test, y_test)

def xgbFull(X_train, X_test, y_train, y_test):
    model = build_xgbModel(X_train, y_train)

    scores = score_xgbModel(model, X_test, y_test)

    preds = model.predict(X_test)

    pred_proba = model.predict_proba(X_test)[:,1]

    metrics = evaluate_model_performance(y_test, preds, pred_proba)

    print(metrics)



    print(scores)


### SVM Model 

def build_svm(features, target, kernel): 
    svm_model = SVC(kernel=kernel, probability=True)
    svm_model.fit(features, target)
    return svm_model


def score_svm(model, X_test, y_test):
    return model.score(X_test, y_test)

def svmFull(X_train, X_test, y_train, y_test, kernel):
    model = build_svm(X_train, y_train, kernel)

    scores = score_svm(model, X_test, y_test)

    preds = model.predict(X_test)

    pred_proba = model.predict_proba(X_test)[:,1]

    metrics = evaluate_model_performance(y_test, preds, pred_proba)

    print(metrics)



    print(scores)


### attempt at FFNN 

def ffnn(features, targets): 
    fut.run_ffnn_model(features,targets, 2)
    

### Logistic Regression 
    
def build_LogReg(features, target, kernel): 
    logreg_model = LogisticRegression()
    logreg_model.fit(features, target)
    return logreg_model


def score_LogReg(model, X_test, y_test):
    return model.score(X_test, y_test)

def LogRegFull(X_train, X_test, y_train, y_test):
    model = build_LogReg(X_train, y_train)

    scores = score_LogReg(model, X_test, y_test)

    preds = model.predict(X_test)

    pred_proba = model.predict_proba(X_test)[:,1]

    metrics = evaluate_model_performance(y_test, preds, pred_proba)

    print(metrics)

    print(scores)

