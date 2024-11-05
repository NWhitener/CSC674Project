import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import posion_utils as pu


def heart(posion, percent, number):
    data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    if posion == "FLIP":
        data2 = pu.flip_random_labels(data=data, percent=percent)
    if posion == "INJECT": 
        data2 = pu.inject_new(data=data, number=number)
    if posion == "TAMPER": 
        data2 = pu.tamper_rows(data= data, percent=percent)
    X = data2.drop(columns = ['Tampered'])
    y = data2['Tampered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=43)
    return X_train, X_test, y_train, y_test