import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_bank_data(df):
    """
    Preprocesses a single bank's dataset for the federated learning setup.
    Includes feature engineering (balance deltas), dropping unnecessary columns,
    one-hot encoding, and standard scaling.
    """
    df = df.copy()
    
    # 1. Create balance delta features
    df['delta_balance_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['delta_balance_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # 2. Drop unwanted columns
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'oldbalanceOrg', 
                    'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    if 'BankID' in df.columns:
        cols_to_drop.append('BankID')
        
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 3. One-hot encode transaction type
    df = pd.get_dummies(df, columns=['type'], drop_first=False)
    
    # 4. Separate Features (X) and Target (y)
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    
    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Fit StandardScaler ONLY on training data
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, y_train, X_test_scaled, y_test
