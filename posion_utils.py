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
    return data_copy


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

#################################################################
# Next Functions are not necessarily posions but data tampering #
# They do not have a tamper label so we can't tell if they are  #
# tampered with but I thought it was interesting to see what    #
# happens                                                       #
#################################################################


def permute_rows(data): 
    data_copy = data.copy()
    data_copy = data_copy.reindex(np.random.permutation(data_copy.index))
    return data_copy 

def permute_columns(data):
    data_copy = data.copy()
    data_copy = data_copy.reindex(columns=np.random.permutation(data_copy.columns))
    return data_copy


def duplicate_columns(data):
    data_copy = data.copy()
    random_column = np.random.randint(0,data_copy.shape[1])
    column_copy = data_copy.iloc[:, random_column]
    data_copy["Column Copy"] = column_copy
    return data_copy






