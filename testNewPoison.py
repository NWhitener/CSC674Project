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

    # Apply misdirection poisoning
    data = poison_utils.misdirection(data, number=10, complexity="major")

    # Separate features and labels
    X = data.drop(columns=['Tampered'])
    y = data['Tampered']

    # Run anomaly detection models
    isoForestDataReturn = mu.build_isolation_forest(X)

    # Check rows where data was tampered
    tampered_data = data[y == 1]  # Select rows where Tampered is 1
    detected_anomalies = isoForestDataReturn.loc[y == 1]

    print("\nDetection results for inserted data:")
    print(detected_anomalies[['anomaly']])  




if __name__ == "__main__":
    main()




