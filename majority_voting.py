import poison_utils as pput 
import model_utils as mut 
import preprocessing_utils as put 
from collections import Counter
import pandas as pd 
 
def detect_poison(data): 
    '''
    Use the three outlier detection methods to determine if a data point is considered to be an anomaly across multiple methods
    If the majority agree that it is an anomaly, report it as being potentially poisoned
    '''
    data2 = data.drop(columns = ['Tampered'])
    data_isolation_forest = mut.build_isolation_forest(data2)
    data_abod = mut.build_abod(data2)
    data_cblof = mut.build_cblof(data2)

    data2['ISOLATION_FOREST'] = data_isolation_forest['anomaly']
    data2['ABOD'] = data_abod['label']
    data2['CBLOF'] = data_cblof['label']
    data2['ISOLATION_FOREST'] = data2["ISOLATION_FOREST"].map({-1:1, 1:0})

    results = []
    for idx, row in data2.iterrows(): 
        vote_1 = row['ISOLATION_FOREST']
        vote_2 = row["ABOD"]
        vote_3 = row["CBLOF"]
        votes = Counter([vote_1,vote_2, vote_3])

        vote = votes.most_common(1)[0][0]  
        results.append(vote)

    data2["Poisoned"] = results
    data2['Tampered'] = data['Tampered']
    print(data2['Tampered'].value_counts())
    return data2            

def main():
    data = pd.read_csv('breast-cancer.csv')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    column_to_move = data.columns[1] 
    data = data[[col for col in data.columns if col != column_to_move] + [column_to_move]]

    data2 = pput.misdirection(data, 10, "major")
    print(len(data2[data2['Tampered']==1]))
    data3 = detect_poison(data2)

    print(data3[data3['Tampered'] == 1]['Poisoned'].value_counts())

    return 

if __name__ == '__main__': 
    main()
