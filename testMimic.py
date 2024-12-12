import pipeline_utils as put
import model_utils as mu
import poison_utils as poison_utils
import pandas as pd

from sklearn.metrics import accuracy_score

def main():
    print("Misdirection")

    # Load and prepare the data
    data = pd.read_csv('breast-cancer.csv')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    column_to_move = data.columns[1] 
    data = data[[col for col in data.columns if col != column_to_move] + [column_to_move]]

    # Apply misdirection poisoning
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

    detected_anomalies_ISO = isoForestDataReturn.loc[y == 1]
    detected_anomalies_ABOD = abodDataReturn.loc[y==1]
    detected_anomalies_cblof = cblofDataReturn.loc[y==1]

    print("\nDetection results for ISO:")
    print(detected_anomalies_ISO[['anomaly']])  

    print("\nDetection results for ABOD data:")
    print(detected_anomalies_ABOD[['label']])  

    print("\nDetection results for cblof data:")
    print(detected_anomalies_cblof[['label']])  




if __name__ == "__main__":
    main()




