import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import poison_utils as pu
from sklearn.preprocessing import OneHotEncoder


def heart(poison, percent, number, mode):
    data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    if poison == "FLIP":
        data2 = pu.flip_random_labels(data=data, percent=percent, dataset='heart')
    if poison == "INJECT": 
        data2 = pu.inject_new(data=data, number=number, mode = mode)
    if poison == "TAMPER": 
        data2 = pu.tamper_rows(data= data, percent=percent, mode = mode)
    X = data2.drop(columns = ['Tampered'])
    y = data2['Tampered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=43)
    return X_train, X_test, y_train, y_test

def loan(poison, percent, number, mode): 
    data = pd.read_csv('loan_data.csv')
    data = pd.get_dummies(data, columns=['person_education', 'person_home_ownership', 'loan_intent','previous_loan_defaults_on_file', 'person_gender'], drop_first=True)
    if poison == "FLIP":
        data2 = pu.flip_random_labels(data=data, percent=percent, dataset='loan')
    if poison == "INJECT": 
        data2 = pu.inject_new(data=data, number=number, mode = mode)
    if poison == "TAMPER": 
        data2 = pu.tamper_rows(data= data, percent=percent, mode = mode )
    X = data2.drop(columns = ['Tampered'])
    y = data2['Tampered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=43)
    return X_train, X_test, y_train, y_test


def cancer(poison, percent, number, mode): 
    data = pd.read_csv('breast-cancer.csv')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    column_to_move = data.columns[1] 
    data = data[[col for col in data.columns if col != column_to_move] + [column_to_move]]
    if poison == "FLIP":
        data2 = pu.flip_random_labels(data=data, percent=percent, dataset='cancer')
    if poison == "INJECT": 
        data2 = pu.inject_new(data=data, number=number, mode=mode)
    if poison == "TAMPER": 
        data2 = pu.tamper_rows(data= data, percent=percent, mode = mode)
    X = data2.drop(columns = ['Tampered'])
    y = data2['Tampered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=43)
    return X_train, X_test, y_train, y_test


def machine(poison, percent, number, mode): 
    data = pd.read_csv('machine_failure_dataset.csv')
    data['Machine_Type'] = data['Machine_Type'].map({'Lathe': 1, 'Drill': 0, 'Mill': 2,})
    if poison == "FLIP":
        data2 = pu.flip_random_labels(data=data, percent=percent, dataset='machine')
    if poison == "INJECT": 
        data2 = pu.inject_new(data=data, number=number, mode=mode)
    if poison == "TAMPER": 
        data2 = pu.tamper_rows(data= data, percent=percent, mode = mode)
    X = data2.drop(columns = ['Tampered'])
    y = data2['Tampered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=43)
    return X_train, X_test, y_train, y_test