import pipeline_utils as put
import model_utils as mu
import poison_utils as poison_utils
import pandas as pd

from sklearn.metrics import accuracy_score

def main():
    # We'll store results from all runs in this list
    all_runs_results = []

    for run_index in range(1, 101):
        # Load and prepare the data
        data = pd.read_csv('loan_data.csv')
        
        #~~~~~~~~~~~~~~~~~~~For Cancer~~~~~~~~~~~~~~
        #data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        #column_to_move = data.columns[1] 
        #data = data[[col for col in data.columns if col != column_to_move] + [column_to_move]]

        #~~~~~~~~~~~~~~~For Loan~~~~~~~~~~~~~~~~~~~~
        data = pd.get_dummies(data, columns=['person_education', 'person_home_ownership', 'loan_intent','previous_loan_defaults_on_file', 'person_gender'], drop_first=True)


        # Apply misdirection poisoning (redo each iteration)
        data = poison_utils.mimic(data, percent=0.05)
        
        # Separate features and labels
        X = data.drop(columns=['Tampered'])
        y = data['Tampered']

        # Run anomaly detection models
        isoForestDataReturn = mu.build_isolation_forest(X)
        abodDataReturn = mu.build_abod(X)
        cblofDataReturn = mu.build_cblof(X)

        # Check rows where data was tampered
        tampered_data = data[y == 1]  # Select rows where Tampered is 1
        total_poison = tampered_data.shape[0]  # How many rows were actually tampered

        # For Isolation Forest, anomalies are -1
        iso_sum = (isoForestDataReturn.loc[y == 1, 'anomaly'] == -1).sum()

        # For ABOD and CBLOF, anomalies are labeled as 1
        abod_sum = abodDataReturn.loc[y == 1, 'label'].sum()
        cblof_sum = cblofDataReturn.loc[y == 1, 'label'].sum()

        methods = ['ISO', 'ABOD', 'CBLOF']
        correct_poison_list = [iso_sum, abod_sum, cblof_sum]

        for method_name, correct_poison in zip(methods, correct_poison_list):
            fraction = correct_poison / total_poison if total_poison != 0 else 0
            all_runs_results.append({
                'Run': run_index,
                'Method': method_name,
                'Correctly Identified Poison': correct_poison,
                'Fraction Correct (Poison)': fraction,
                'Total Poison': total_poison
            })
    
    # After 100 runs, create a single DataFrame with all results
    df_all_runs = pd.DataFrame(all_runs_results)

    # Write the entire results to one CSV
    df_all_runs.to_csv('detection_results_100_runs.csv', index=False)
    print("All run results saved to detection_results_100_runs.csv.")

if __name__ == "__main__":
    main()
