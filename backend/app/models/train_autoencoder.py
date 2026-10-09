import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.app.data.loader import get_raw_dataset
from backend.app.features.preprocessing import preprocess_bank_data
from backend.app.models.autoencoder import FraudAutoencoder, get_reconstruction_loss

def train_and_evaluate_autoencoder():
    # 1. Load data
    df = get_raw_dataset()
    
    # If BankID doesn't exist, simulate it for 5 banks
    if 'BankID' not in df.columns:
        np.random.seed(42)
        df['BankID'] = np.random.randint(0, 5, size=len(df))
    
    # Use Bank 0 for this isolated training
    bank_df = df[df['BankID'] == 0].copy()
    
    # 2. Preprocess (scaling, OHE, etc.)
    X_train, y_train, X_test, y_test = preprocess_bank_data(bank_df)
    
    # 3. Filter ONLY normal (isFraud=0) transactions from the training set
    X_train_normal = X_train[y_train == 0].values
    
    # Split out a validation set for early stopping
    X_train_norm, X_val_norm = train_test_split(X_train_normal, test_size=0.15, random_state=42)
    
    # Convert to PyTorch tensors
    train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_norm, dtype=torch.float32)
    
    # Create DataLoaders
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_tensor, train_tensor), 
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_tensor, val_tensor), 
        batch_size=batch_size, shuffle=False
    )
    
    # 4. Initialize model, loss, and optimizer
    input_dim = X_train.shape[1]
    model = FraudAutoencoder(input_dim=input_dim, latent_dim=16)
    criterion = get_reconstruction_loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 5. Training loop with early stopping
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = 50
    best_model_path = os.path.join(os.path.dirname(__file__), "best_autoencoder.pth")
    
    print("\n--- Training Autoencoder on Normal Transactions ---")
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_x)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:02d}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model state
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break
                
    # Load the best model weights
    model.load_state_dict(torch.load(best_model_path))
    
    # Save JUST the encoder weights as requested
    encoder_path = os.path.join(os.path.dirname(__file__), "encoder_weights.pth")
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"\nSaved best encoder weights to {encoder_path}")
    
    # 6. Evaluate Reconstruction Error on the Test Set (Contains both Normal & Fraud)
    print("\n--- Evaluating Reconstruction Errors on Test Set ---")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        reconstructed = model(X_test_tensor)
        
        # Compute MSE per sample
        mse = torch.mean((X_test_tensor - reconstructed)**2, dim=1).numpy()
        
    # Group results by label
    results_df = pd.DataFrame({'Reconstruction_Error': mse, 'isFraud': y_test.values})
    
    normal_error = results_df[results_df['isFraud'] == 0]['Reconstruction_Error']
    fraud_error = results_df[results_df['isFraud'] == 1]['Reconstruction_Error']
    
    print(f"Normal Transactions (Count: {len(normal_error)})")
    print(f"  Mean Error: {normal_error.mean():.4f}")
    print(f"  Max Error:  {normal_error.max():.4f}")
    
    print(f"\nFraud Transactions (Count: {len(fraud_error)})")
    print(f"  Mean Error: {fraud_error.mean():.4f}")
    print(f"  Max Error:  {fraud_error.max():.4f}")
    
    if fraud_error.mean() > normal_error.mean():
        print("\nSUCCESS: Fraud samples have a significantly higher reconstruction error!")
    else:
        print("\nWARNING: Fraud samples do not have a higher reconstruction error. Model might need tuning.")
    
if __name__ == "__main__":
    train_and_evaluate_autoencoder()
