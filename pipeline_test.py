import model_utils as mods 
import poison_utils as pu 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pipeline_utils as put

def main(): 
    print("Heart Dataset")
    put.test_posion(posionType='FLIP', percent=0.1, number=0, dataset='heart')
    print("Cancer Dataset")
    put.test_posion(posionType='FLIP', percent=0.1, number = 0, dataset='cancer')
    


if __name__ == '__main__': 
    main()