import pandas as pd

def print_averages(csv_path):
    df = pd.read_csv(csv_path)
    grouped = df.groupby('Method').agg({
        'Correctly Identified Poison': 'mean',
        'Fraction Correct (Poison)': 'mean',
        'Total Poison': 'mean'
    })

    # Round or format the floats
    grouped = grouped.round(3)

    # Print out row by row
    for method, row in grouped.iterrows():
        print(f"Method: {method}")
        print(f"  Avg Correctly Identified Poison: {row['Correctly Identified Poison']}")
        print(f"  Avg Fraction Correct (Poison):   {row['Fraction Correct (Poison)']}")
        print(f"  Avg Total Poison:               {row['Total Poison']}")
        print()

if __name__ == '__main__':
    print_averages('misdirectionCancer.csv')
