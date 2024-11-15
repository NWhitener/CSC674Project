from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras import optimizers
import time
import sys
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
import tracemalloc

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score
import pandas as pd

def evaluate_nmi(observed, predicted):
   nmi = normalized_mutual_info_score(observed, predicted)
   return(nmi)

def evaluate_roc(observed, predicted):
   roc = roc_auc_score(observed, predicted)
   return(roc)

def evaluate_kappa(observed, predicted):
   kap = cohen_kappa_score(observed, predicted)
   return(kap)

def evaluate_f1(observed, predicted):
   f1 = precision_recall_fscore_support(observed, predicted, average='weighted')
   return(f1)

def create_ffnn_model(train_x, train_y, test_x, num_labels):
    model = Sequential()
    model.add(Dense(256, input_dim=train_x.shape[1], kernel_regularizer=l2(0.01), activation="relu"))
    model.add(Dropout(0.7))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_labels,activation = "sigmoid"))
    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['binary_accuracy'])
    return(model)

def train_ffnn(model,train_x, train_y):
    model.fit(train_x, train_y, epochs = 50, batch_size = 128, verbose =1)
    (loss, accuracy) = model.evaluate(train_x, train_y, batch_size=128, verbose=1)
    return(model)

def test_ffnn(model, test_x):
    predictions = np.argmax(model.predict(test_x), axis=-1)
    return(predictions)

  
def run_ffnn_model(data, labels, num_labels):
    results = pd.DataFrame(columns=['FFNN_NMI_AVE', 'FFNN_NMI_STD', 'FFNN_ROC_AVE','FFNN_ROC_STD','FFNN_KAPPA_AVE', 'FFNN_KAPPA_STD'])
    fold_number = 1
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    nmi_scores_ffnn = []
    auc_scores_ffnn = []
    kappa_scores_ffnn = []
    f1_scores_ffnn = []
    for train, test in kfold.split(data, np.argmax(labels, axis= -1)):
        fold_result = pd.DataFrame()
        fold_result['class'] = np.argmax(labels[test], axis= -1)
        fold_result['FFNN'] = train_ffnn(data[train], labels[train], data[test], num_labels)
        nmi_scores_ffnn.append(evaluate_nmi(fold_result['class'], fold_result['FFNN']))
        auc_scores_ffnn.append(evaluate_roc(fold_result['class'], fold_result['FFNN']))
        kappa_scores_ffnn.append(evaluate_kappa(fold_result['class'], fold_result['FFNN']))
        f1_scores_ffnn.append(evaluate_f1(fold_result['class'], fold_result['FFNN']))
        fold_number = fold_number + 1
    results = results.append({
              'FFNN_NMI_AVE': np.mean(nmi_scores_ffnn),
              'FFNN_NMI_STD': np.std(nmi_scores_ffnn),
              'FFNN_ROC_AVE': np.mean(auc_scores_ffnn),
              'FFNN_ROC_STD': np.std(auc_scores_ffnn),
              'FFNN_KAPPA_AVE': np.mean(kappa_scores_ffnn),
              'FFNN_KAPPA_STD': np.std(kappa_scores_ffnn),
              },
              ignore_index=True)
    results.to_csv('Metrics_3layer_binary_acc_sig.csv')