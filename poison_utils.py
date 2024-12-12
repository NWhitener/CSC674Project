'''
This file will be used to write the methods that 
we need tp effectively poison the data set
'''
import pandas as pd 
import numpy as np 
import random as rand


def prep_poision(data): 
    data['Tampered'] = 0 
    return data

'''
This function randomly flipps the label a certain percent of the dataset. The rate of fliping can be variable
Could explore how this is applied? Different random selection? etc. 
'''

#~~~~~~~~~~~~~~~~~ Flips ~~~~~~~~~~~~~~~~~
def flip_random_labels(data, percent, dataset):
    
    data_copy = data.copy()
    #Figure out how much to tamper with
    total = int(percent * len(data))
    #Select a random sample of the dataset
    flips = np.random.randint(0, len(data), size=total)
    data_copy = prep_poision(data_copy)
    if dataset == 'heart': 
        for i in flips: 
            if data_copy.loc[i, 'DEATH_EVENT'] == 0: 
                #Flip the data point
                data_copy.loc[i, 'DEATH_EVENT'] = 1
                data_copy.loc[i, 'Tampered'] = 1
            else: 
                data_copy.loc[i, 'DEATH_EVENT'] = 0
                data_copy.loc[i,'Tampered'] = 1
    if dataset == 'cancer': 
        for i in flips: 
            if data_copy.loc[i, 'diagnosis'] == 0: 
                #Flip the data point
                data_copy.loc[i, 'diagnosis'] = 1
                data_copy.loc[i, 'Tampered'] = 1
            else: 
                data_copy.loc[i, 'diagnosis'] = 0
                data_copy.loc[i,'Tampered'] = 1   
    return data_copy

#~~~~~~~~~~~~~ Inject ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def inject_new(data, number, mode): 
    copy_data = data.copy()  
    print(f'Original length: {len(copy_data)}')

    copy_data = prep_poision(copy_data)

    #Inject completely random data
    if mode == 'random': 
        #Inject the specified number of rows
        for i in range(number):
            new_data = []

            for column in copy_data.columns[:-1]: 
                #Randomly generate a new piece of data based on the max of all of the values in a particular column
                random_val_1 = int(np.random.uniform(0, data[column].max()))
                new_data.append(random_val_1)
            #Mark it as tampered
            new_data.append(1)
            #Add it to the dataset
            copy_data.loc[len(copy_data)] = new_data   
    
    elif mode == 'distribution': 
        for i in range(number):
            new_data = []

            for column in copy_data.columns[:-1]: 
                #Randomly generate a new piece of data based on the normal distribution of all values in the dataset
                random_val_1 = int(np.random.normal(loc=data[column].mean(), scale=data[column].std()) ) 
                new_data.append(random_val_1)
            #Mark it as tampered
            new_data.append(1)
            #Add it to the dataset
            copy_data.loc[len(copy_data)] = new_data   
    
    elif mode == 'malicious': 
        for i in range(number):
            new_data = []

            for column in copy_data.columns[:-1]: 
                #Randomly generate a new piece of data based on absolutley nothing
                random_val_1 = int(np.random.uniform(-10000, 10000))
                new_data.append(random_val_1)
            #Mark it as tampered
            new_data.append(1)
            #Add it to the dataset
            copy_data.loc[len(copy_data)] = new_data   

    print(f'New length after injection: {len(copy_data)}')
    return copy_data

#~~~~~~~~~~~~~Tamper~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#~~~~~~~~~~~~~~~~~~~~Misdirection~~~~~~~~~~~~~~~~~
def misdirection(data, number,complexity):
    data_copy = data.copy()
    data_copy = prep_poision(data_copy)

    #Number of purposefully bad added rows (should be 25%)
    fakeNum = number // 4
    #Number of realistic added rows (should be 75%)
    realNum = number - fakeNum

    #Fake Added rows (purposefully bad)
    for i in range(fakeNum):
        new_data = []

        for column in data_copy.columns[:-1]:  #exclude the last column
            #Generate obviously fake data using 3 times the standard deviation
            random_val_1 = int(np.random.normal(loc=data[column].mean(), scale= 2 * data[column].std()))
            new_data.append(random_val_1)

        #Mark it as tampered
        new_data.append(1)  #Append tampered label
        #Add it to the dataset
        data_copy.loc[len(data_copy)] = new_data

    #Realistic rows (copying a row, changing one column by std)
    #~~~~~~~~~~~~~~~~~~~~~~~~one Col by STD~~~~~~~~~~~~~~~~~~~~~~~~
    if complexity == "minor":
        print("Minor version (add rows based on mean and std)")

        #COPYING A ROW, CHANGING ONE COLUMN BY STD
        for i in range(realNum):
            #choose a random column to modify (excluding the last column)
            column_to_modify = np.random.randint(0, data_copy.shape[1] - 1)

            #get column statistics
            mean = data.iloc[:, column_to_modify].mean()
            std = data.iloc[:, column_to_modify].std()

            #choose random row index to copy
            random_row_index = np.random.randint(data_copy.shape[0])
            random_row = data_copy.iloc[random_row_index].copy()  # Copy the row

            #modify the selected column slightly
            random_row.iloc[column_to_modify] = int(random_row[column_to_modify] + std)
            
            #mark the row as tampered
            random_row['Tampered'] = 1
            #add the modified row to the dataset
            data_copy.loc[len(data_copy)] = random_row
        
    #~~~~~~~~~~~~~~~~~~~~~everything mean+std deviation~~~~~~~~~~~~~~~~~~~~~~~
    if complexity == "major":
        print("Major version add rows based on mean and std")

        for i in range(realNum):    
            new_data = []

            for column in data_copy.columns[:-1]:  #Exclude the last column
                #generate data using the column's mean and std
                random_val_1 = np.random.normal(loc=data[column].mean(), scale=data[column].std())
                new_data.append(random_val_1)

            #mark it as tampered
            new_data.append(1)  # Append tampered label
            #Add it to the dataset
            data_copy.loc[len(data_copy)] = new_data
    
    #~~~~~~~~~~~~~~~~~~Random value from the column~~~~~~~~~~~~~~~~~~
    if complexity == "random":
        print("random version: add rows based on random values from existing rows")

        for i in range(realNum):    
            new_data = []

            for column in data_copy.columns[:-1]:  #Exclude the last column
                #select a random value from the current column
                random_val_1 = data[column].sample(n=1).values[0]
                new_data.append(random_val_1)

            #Mark it as tampered
            new_data.append(1)  #append tampered label
            #Add it to the dataset
            data_copy.loc[len(data_copy)] = new_data


    return data_copy


#~~~~~~~~~~~~~~~~~~~Mimic Chest~~~~~~~~~~~

def mimic(data, percent):
    print("AHHHH A mimic")

    data_copy = data.copy()
    data_copy = prep_poision(data_copy)

    #Determine indices for columns:
    #Assuming 'diagnosis' is second to last column and 'Tampered' is last
    diagnosis_col_idx = -2
    tampered_col_idx = -1

    #Calculate how many rows to tamper
    num_rows_to_modify = int(percent * len(data_copy))

    for _ in range(num_rows_to_modify):
        #choose random row
        random_row_index = np.random.randint(data_copy.shape[0])


        #print(data_copy.iloc[random_row_index])


        #Get original label
        original_label = data_copy.iloc[random_row_index, diagnosis_col_idx]
        
        #Dtermine new label (flip it)
        new_label = 1 if original_label == 0 else 0

        #set new diagnosis label
        data_copy.iloc[random_row_index, diagnosis_col_idx] = new_label

        # Choose rows that have the new_label for feature copying
        # For each column except the last two, we find a row with the target label (new_label)
        for column in data_copy.columns[:-2]:
            notFound = True
            while notFound:
                candidate_idx = np.random.randint(data_copy.shape[0])
                if data_copy.iloc[candidate_idx, diagnosis_col_idx] == new_label:
                    #Copy the column value from the candidate to the chosen row
                    data_copy.at[random_row_index, column] = data_copy.iat[candidate_idx, data_copy.columns.get_loc(column)]
                    #We found a candidate for this column, break the while loop
                    notFound = False

        #Set Tampered to 1
        data_copy.iloc[random_row_index, tampered_col_idx] = 1

        #Print  modified row
        #print(data_copy.iloc[random_row_index])

    return data_copy


#################################################################
# Next Functions are not necessarily poisons but data tampering #
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






