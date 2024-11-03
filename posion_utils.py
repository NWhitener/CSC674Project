'''
This file will be used to write the methods that 
we need tp effectively posion the data set
'''

'''
Challenges that we need to consider: How 
'''


import pandas as pd 
import numpy as np 


data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

def prep_poision(data): 
    data['Tampered'] = 0 
    return data

'''
This function randomly flipps the label a certain percent of the dataset. The rate of fliping can be variable
Could explore how this is applied? Different random selection? etc. 
'''
def flip_random_labels(data, percent): 
    #Figure out how much to tamper with
    total = int(percent * len(data))
    print(total)
    #Select a random sample of the dataset
    flips = np.random.randint(0, len(data), size=total)
    print(flips)
    data = prep_poision(data)
    for i in flips: 
        print(data.loc[i, 'DEATH_EVENT'])
        if data.loc[i, 'DEATH_EVENT'] == 0: 
            #Flip the data point
            data.loc[i, 'DEATH_EVENT'] = 1
            data.loc[i, 'Tampered'] = 1
        else: 
            data.loc[i, 'DEATH_EVENT'] = 0
            data.loc[i,'Tampered'] = 1
    return data

def inject_new(data, number): 
    data = prep_poision(data)
    


def main(): 
    data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    data2 = flip_random_labels(data, 0.05)
    print(data2)
    print(data2['Tampered'].value_counts())


if __name__ == '__main__': 
    main()