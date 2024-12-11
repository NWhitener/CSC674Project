import preprocessing_utils as put 
import model_utils as mut 
import poison_utils as pput
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import pandas as pd 
import majority_voting as mv 
import preprocessing_utils as ppu 
import pipeline_utils as pu

data = ppu.heart_load()

data2 = pu.test_poison_demo(data, 'TAMPER', 'HEART')

print(data2)