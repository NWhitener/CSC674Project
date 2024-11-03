'''
This file will be used to write the methods that 
we need tp effectively posion the data set
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
    
    data_copy = data.copy()
    print(len(data_copy)) 
    #Figure out how much to tamper with
    total = int(percent * len(data))
    #Select a random sample of the dataset
    flips = np.random.randint(0, len(data), size=total)
    data_copy = prep_poision(data_copy)
    for i in flips: 
        if data_copy.loc[i, 'DEATH_EVENT'] == 0: 
            #Flip the data point
            data_copy.loc[i, 'DEATH_EVENT'] = 1
            data_copy.loc[i, 'Tampered'] = 1
        else: 
            data_copy.loc[i, 'DEATH_EVENT'] = 0
            data_copy.loc[i,'Tampered'] = 1
    print(len(data_copy))
    return data_copy

import pandas as pd
import numpy as np

def inject_new(data, number, mode): 
    copy_data = data.copy()  
    print(f'Original length: {len(copy_data)}')

    copy_data = prep_poision(copy_data)

    # Inject completely random data
    if mode == 'random': 
        # Inject the specified number of rows
        for i in range(number):
            new_data = []

            for column in copy_data.columns[:-1]: 
                # Randomly generate a new piece of data based on the max of all of the values in a particular column
                random_val_1 = int(np.random.uniform(0, data[column].max()))
                new_data.append(random_val_1)
            # Mark it as tampered
            new_data.append(1)
            # Add it to the dataset
            copy_data.loc[len(copy_data)] = new_data   
    
    elif mode == 'distribution': 
        for i in range(number):
            new_data = []

            for column in copy_data.columns[:-1]: 
                # Randomly generate a new piece of data based on the normal distribution of all values in the dataset
                random_val_1 = int(np.random.normal(loc=data[column].mean(), scale=data[column].std()) ) 
                new_data.append(random_val_1)
            # Mark it as tampered
            new_data.append(1)
            # Add it to the dataset
            copy_data.loc[len(copy_data)] = new_data   
    
    elif mode == 'malicious': 
        for i in range(number):
            new_data = []

            for column in copy_data.columns[:-1]: 
                # Randomly generate a new piece of data based on absolutley nothing
                random_val_1 = int(np.random.uniform(-10000, 10000))
                new_data.append(random_val_1)
            # Mark it as tampered
            new_data.append(1)
            # Add it to the dataset
            copy_data.loc[len(copy_data)] = new_data   

    print(f'New length after injection: {len(copy_data)}')
    return copy_data

def tamper_rows(data, percent, mode): 
    data_copy = data.copy()
    data_copy = prep_poision(data_copy)
    total = int(percent*len(data_copy))
    rows = np.random.randint(0, len(data), size=total)

    if mode == 'random': 
        for i in rows: 
            for column in data_copy.columns[:-1]: 
                data_copy.loc[i, column] = int(np.random.uniform(0, data[column].max()))
            data_copy.loc[i,'Tampered'] = 1  

    if mode == "distribution": 
        for i in rows: 
            for column in data_copy.columns[:-1]: 
                data_copy.loc[i, column] = int(np.random.normal(loc=data[column].mean(), scale=data[column].std())  )
            data_copy.loc[i,'Tampered'] = 1
    if mode == 'malicious': 
        for i in rows: 
            for column in data_copy.columns[:-1]: 
                data_copy.loc[i, column] = int(np.random.uniform(-10000,10000))
            data_copy.loc[i,'Tampered'] = 1
    return data_copy   

def main(): 
    data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    data.drop(columns=['time'], inplace = True)
    data2 = flip_random_labels(data, 0.05)
    data3 = inject_new(data, 14, 'random')
    data4 = inject_new(data, 14, 'distribution')
    data5 = inject_new(data, 14, 'malicious')
    data6 = tamper_rows(data, 0.05, 'random')
    data7 = tamper_rows(data, 0.05, 'distribution')
    data8 = tamper_rows(data, 0.05,'malicious')
    print(data2["Tampered"].value_counts())
    print(data3["Tampered"].value_counts())
    print(data4["Tampered"].value_counts())
    print(data5["Tampered"].value_counts())
    print(data6["Tampered"].value_counts())
    print(data7["Tampered"].value_counts())
    print(data8["Tampered"].value_counts())
    
if __name__ == '__main__': 
    main()