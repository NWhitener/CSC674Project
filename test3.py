import preprocessing_utils as put 
import model_utils as mut 
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd 

heart = pd.read_csv("/Users/nathanwhitener/Desktop/Machine Learning/Project/CSC674Project/heart_failure_clinical_records_dataset.csv")
print(heart['DEATH_EVENT'].value_counts())
loan = pd.read_csv("/Users/nathanwhitener/Desktop/Machine Learning/Project/CSC674Project/loan_data.csv")
print(loan["loan_status"].value_counts())
print(loan.shape)
machine = pd.read_csv("/Users/nathanwhitener/Desktop/Machine Learning/Project/CSC674Project/machine_failure_dataset.csv")
print(machine['Failure_Risk'].value_counts())
cancer = pd.read_csv("/Users/nathanwhitener/Desktop/Machine Learning/Project/CSC674Project/breast-cancer.csv")
print(cancer['diagnosis'].value_counts())