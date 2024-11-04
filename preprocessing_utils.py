import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import posion_utils as pu


def heart():
    data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    data2 = pu.flip_random_labels(data=data, percent=0.10)
    X = data2.drop(columns = ['Tampered'])
    y = data2['Tampered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=43)
    return X_train, X_test, y_train, y_test