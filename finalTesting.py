import poison_utils as pput
import model_utils as mut
import preprocessing_utils as put
import pandas as pd
from collections import Counter
import random

def majority_voting_100_times(data, poison_type, **kwargs):
    """
    Apply majority voting 100 times for the specified poison type.

    Args:
        data (DataFrame): The dataset to poison and evaluate.
        poison_type (str): Type of poison (e.g., "misdirection", "mimic").
        kwargs: Additional arguments to pass to the poisoning method.

    Returns:
        results (list): A list of results from each run.
    """
    results = []
    
    for _ in range(100):
        poisoned_data = getattr(pput, poison_type)(data.copy(), **kwargs)
        result = detect_poison(poisoned_data)
        results.append(result['Poisoned'].value_counts())

    return results

def detect_poison(data):
    """
    Use the three outlier detection methods to determine if a data point is considered to be an anomaly across multiple methods.
    If the majority agree that it is an anomaly, report it as being potentially poisoned.

    Args:
        data (DataFrame): The dataset to evaluate.

    Returns:
        DataFrame: The dataset with poison detection results.
    """
    data2 = data.drop(columns=['Tampered'])
    data_isolation_forest = mut.build_isolation_forest(data2)
    data_abod = mut.build_abod(data2)
    data_cblof = mut.build_cblof(data2)

    data2['ISOLATION_FOREST'] = data_isolation_forest['anomaly']
    data2['ABOD'] = data_abod['label']
    data2['CBLOF'] = data_cblof['label']
    data2['ISOLATION_FOREST'] = data2["ISOLATION_FOREST"].map({-1: 1, 1: 0})

    results = []
    for idx, row in data2.iterrows():
        votes = Counter([row['ISOLATION_FOREST'], row['ABOD'], row['CBLOF']])
        vote = votes.most_common(1)[0][0]
        results.append(vote)

    data2["Poisoned"] = results
    data2['Tampered'] = data['Tampered']
    return data2

def main():
    datasets = ["breast-cancer.csv", "heart_failure_clinical_records_dataset.csv", "loan_data.csv", "machine_failure_dataset.csv"]
    poison_types = ["misdirection", "mimic"]
    misdirection_complexities = ["minor", "major", "random"]

    for dataset_path in datasets:
        print(f"Processing dataset: {dataset_path}")
        data = pd.read_csv(dataset_path)

        if 'diagnosis' in data.columns:
            data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
            column_to_move = data.columns[1]
            data = data[[col for col in data.columns if col != column_to_move] + [column_to_move]]

        for poison_type in poison_types:
            if poison_type == "misdirection":
                for complexity in misdirection_complexities:
                    print(f"Testing poison type: {poison_type} with complexity: {complexity}")
                    results = majority_voting_100_times(data, poison_type, number=10, complexity=complexity)
                    print(f"Results for {poison_type} ({complexity}):", results)
            elif poison_type == "mimic":
                print(f"Testing poison type: {poison_type}")
                results = majority_voting_100_times(data, poison_type, percent=0.05)
                print(f"Results for {poison_type}:", results)

if __name__ == '__main__':
    main()
