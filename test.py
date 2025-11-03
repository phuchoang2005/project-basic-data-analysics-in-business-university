import pandas as pd

# Load datasets
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Inspect train.csv
    print("--- Train.csv Info ---")
    train_df.info()
    print("\n--- Train.csv Head ---")
    print(train_df.head())

    # Inspect test.csv
    print("\n--- Test.csv Info ---")
    test_df.info()
    print("\n--- Test.csv Head ---")
    print(test_df.head())

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure 'train.csv' and 'test.csv' are in the correct directory.")