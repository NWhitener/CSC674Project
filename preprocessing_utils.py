import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import poison_utils as pu


def heart(poison, percent, number):
    data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    if poison == "FLIP":
        data2 = pu.flip_random_labels(data=data, percent=percent, dataset='heart')
    if poison == "INJECT": 
        data2 = pu.inject_new(data=data, number=number)
    if poison == "TAMPER": 
        data2 = pu.tamper_rows(data= data, percent=percent)
    X = data2.drop(columns = ['Tampered'])
    y = data2['Tampered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=43)
    return X_train, X_test, y_train, y_test

def loan(poison, percent, number): 
    data = pd.read_csv('loan_data.csv')
    if poison == "FLIP":
        data2 = pu.flip_random_labels(data=data, percent=percent, dataset='loan')
    if poison == "INJECT": 
        data2 = pu.inject_new(data=data, number=number)
    if poison == "TAMPER": 
        data2 = pu.tamper_rows(data= data, percent=percent)
    X = data2.drop(columns = ['Tampered'])
    y = data2['Tampered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=43)
    return X_train, X_test, y_train, y_test


def cancer(poison, percent, number): 
    data = pd.read_csv('breast-cancer.csv')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    if poison == "FLIP":
        data2 = pu.flip_random_labels(data=data, percent=percent, dataset='cancer')
    if poison == "INJECT": 
        data2 = pu.inject_new(data=data, number=number)
    if poison == "TAMPER": 
        data2 = pu.tamper_rows(data= data, percent=percent)
    X = data2.drop(columns = ['Tampered'])
    y = data2['Tampered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=43)
    return X_train, X_test, y_train, y_test