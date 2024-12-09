import preprocessing_utils as put 
import model_utils as mut 
import poison_utils as pput
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import pandas as pd 

data = pd.read_csv('breast-cancer.csv')
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
column_to_move = data.columns[1] 
data = data[[col for col in data.columns if col != column_to_move] + [column_to_move]]

data2 = pput.tamper_rows(data= data, percent=.15, mode = "DISTRIBUTION")


data3 = data2.drop(columns = ["Tampered"])

new_data = mut.build_isolation_forest(data3)

new_data["Tampered"] = data2['Tampered']
print(new_data[new_data["Tampered"] == 1]['anomaly'].value_counts())

new_data_abod = mut.build_abod(data3)

new_data_abod["Tampered"] = data2['Tampered']

print(new_data_abod[new_data_abod["Tampered"] == 1]['label'].value_counts())

new_data_cblof = mut.build_cblof(data3)

new_data_cblof["Tampered"] = data2['Tampered']


print(new_data_cblof[new_data_cblof["Tampered"] == 1]['label'].value_counts())