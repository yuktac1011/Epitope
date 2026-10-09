import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve, 
    auc, 
    roc_curve, 
    precision_score, 
    recall_score, 
    f1_score
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.app.data.loader import get_raw_dataset
from backend.app.features.preprocessing import preprocess_bank_data
from backend.app.models.autoencoder import FraudAutoencoder
from backend.app.models.hybrid_model import HybridFraudModel

def train_and_evaluate_hybrid():
    # 1. Load and prep data for Bank 0
    df = get_raw_dataset()
    if 'BankID' not in df.columns:
        np.random.seed(42)
        df['BankID'] = np.random.randint(0, 5, size=len(df))
        
    bank_id = 0
    bank_df = df[df['BankID'] == bank_id].copy()
    
    # 2. Preprocess
    X_train, y_train, X_test, y_test = preprocess_bank_data(bank_df)
    
    # 3. Create PyTorch Tensors
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    # Calculate pos_weight for class imbalance: (total negatives / total positives)
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    print(f"Calculated pos_weight for class imbalance: {pos_weight.item():.2f}")
    
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t), 
        batch_size=batch_size, shuffle=True
    )
    
    # 4. Load Autoencoder & Create Hybrid Model
    input_dim = X_train.shape[1]
    latent_dim = 16
    
    autoencoder = FraudAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    ae_weights_path = os.path.join(os.path.dirname(__file__), "best_autoencoder.pth")
    if os.path.exists(ae_weights_path):
        autoencoder.load_state_dict(torch.load(ae_weights_path))
        print("Successfully loaded pre-trained autoencoder weights.")
    else:
        print("WARNING: Pre-trained autoencoder not found. Using randomly initialized encoder weights.")
        
    # The hybrid model automatically freezes the encoder internally
    hybrid_model = HybridFraudModel(
        encoder=autoencoder.encoder, 
        input_dim=input_dim, 
        latent_dim=latent_dim, 
        concat_input=True
    )
    
    # 5. Define Loss & Optimizer (Optimizing ONLY the MLP head)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(hybrid_model.get_trainable_parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 6. Training loop
    epochs = 15
    print(f"\n--- Training Hybrid Model on Bank {bank_id} ---")
    for epoch in range(epochs):
        hybrid_model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = hybrid_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss (BCE): {train_loss:.4f}")
        
    # 7. Evaluate
    print("\n--- Evaluating Hybrid Model ---")
    hybrid_model.eval()
    with torch.no_grad():
        test_logits = hybrid_model(X_test_t)
        test_probs = torch.sigmoid(test_logits).squeeze().numpy()
        y_true = y_test.values
        test_preds = (test_probs >= 0.5).astype(int)
        
    # Compute metrics
    ap = average_precision_score(y_true, test_probs)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, test_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    fpr, tpr, _ = roc_curve(y_true, test_probs)
    valid_fpr_indices = np.where(fpr <= 0.01)[0]
    recall_at_1_fpr = tpr[valid_fpr_indices[-1]] if len(valid_fpr_indices) > 0 else 0.0
    
    print(f"Average Precision: {ap:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Recall @ 1% FPR: {recall_at_1_fpr:.4f}")
    print(f"Precision: {precision_score(y_true, test_preds, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, test_preds, zero_division=0):.4f}")
    print(f"F1: {f1_score(y_true, test_preds, zero_division=0):.4f}")
    
    # 8. Compare with Baseline
    baseline_path = os.path.join(os.path.dirname(__file__), "local_baseline_results.csv")
    if os.path.exists(baseline_path):
        base_df = pd.read_csv(baseline_path)
        base_stats = base_df[base_df['BankID'] == bank_id].iloc[0]
        
        print(f"\n--- Comparison vs Logistic Regression Baseline (Bank {bank_id}) ---")
        
        diff_ap = ap - base_stats['Average Precision']
        diff_pr = pr_auc - base_stats['PR-AUC']
        diff_rec = recall_at_1_fpr - base_stats['Recall @ 1% FPR']
        
        print(f"{'Metric':<16} | {'Hybrid (MLP)':<12} | {'LR Baseline':<12} | {'Improvement':<12}")
        print("-" * 60)
        print(f"{'Avg Precision':<16} | {ap:<12.4f} | {base_stats['Average Precision']:<12.4f} | {diff_ap:+.4f}")
        print(f"{'PR-AUC':<16} | {pr_auc:<12.4f} | {base_stats['PR-AUC']:<12.4f} | {diff_pr:+.4f}")
        print(f"{'Recall @ 1% FPR':<16} | {recall_at_1_fpr:<12.4f} | {base_stats['Recall @ 1% FPR']:<12.4f} | {diff_rec:+.4f}")
    else:
        print("\nNote: local_baseline_results.csv not found to run comparisons.")
        
if __name__ == "__main__":
    train_and_evaluate_hybrid()
