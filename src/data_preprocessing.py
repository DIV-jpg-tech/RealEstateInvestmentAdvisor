print("FILE IS RUNNING")
import pandas as pd
from config import RAW_DATA_PATH

def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df
def inspect_data(df):
    print("\n First 5 Rows:")
    print(df.head())

    print("\nShape of Dataset:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nStatistical Summary:")
    print(df.describe())

if __name__ == "__main__":
        df = load_data()
        inspect_data(df)
