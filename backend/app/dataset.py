import os
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_prep_data():
    print("1. Loading dataset from Hugging Face...")
    # Load the dataset
    cache_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    ds = load_dataset("purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection", cache_dir=cache_dir)
    
    # Convert to Pandas DataFrame for easy manipulation
    # Note: Hugging Face usually puts everything in the 'train' split by default
    df = ds['train'].to_pandas()
    
    # HACKATHON TRICK: The dataset is HUGE (~6M rows). 
    # Let's take a sample so your demo runs fast, but ensure we keep fraud cases!
    fraud_df = df[df['isFraud'] == 1]
    legit_df = df[df['isFraud'] == 0].sample(100000, random_state=42) 
    df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42) # Shuffle

    print(f"2. Processing {len(df)} transactions...")

    # -- PREPROCESSING --
    
    # A. Encode the transaction 'type' (CASH_IN, CASH_OUT, TRANSFER, etc.) into numbers
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])
    
    # B. Drop columns that are strings/IDs (Models can't read customer names easily)
    # We also drop 'isFlaggedFraud' as it's a useless column in this dataset
    df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])
    
    # C. Separate Features (X) and Target (y)
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    
    # D. Scale the numbers (Very important for Machine Learning)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def create_federated_banks(X, y):
    print("3. Partitioning data for 3 Banks (Federated Learning Simulation)...")
    
    # Split data into Bank A (33%), and Temp (67%)
    X_bank_A, X_temp, y_bank_A, y_temp = train_test_split(X, y, test_size=0.66, random_state=42)
    
    # Split Temp into Bank B (33%) and Bank C (33%)
    X_bank_B, X_bank_C, y_bank_B, y_bank_C = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Bank A: {len(X_bank_A)} transactions")
    print(f"Bank B: {len(X_bank_B)} transactions")
    print(f"Bank C: {len(X_bank_C)} transactions")
    
    return (X_bank_A, y_bank_A), (X_bank_B, y_bank_B), (X_bank_C, y_bank_C)

# --- RUN THE SCRIPT ---
if __name__ == "__main__":
    # Get cleaned data
    X, y = load_and_prep_data()
    
    # Get simulated banks
    bank_a_data, bank_b_data, bank_c_data = create_federated_banks(X, y)
    
    print("Data is ready for Federated Learning!")