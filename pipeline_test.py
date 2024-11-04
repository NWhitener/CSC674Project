import model_utils as mods 
import posion_utils as pu 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import preprocessing_utils as put

def main(): 
    X_train, X_test, y_train, y_test = put.heart()
    mods.xgbFull(X_train, X_test, y_train, y_test)




if __name__ == '__main__': 
    main()