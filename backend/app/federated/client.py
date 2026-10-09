import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, roc_curve

# Ensure the backend directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.app.features.preprocessing import preprocess_bank_data
from backend.app.models.autoencoder import FraudAutoencoder
from backend.app.models.hybrid_model import HybridFraudModel

def get_parameters(net) -> List[np.ndarray]:
    """Extract only the trainable parameters (MLP head) from the model."""
    return [val.cpu().detach().numpy() for name, val in net.named_parameters() if val.requires_grad]

def set_parameters(net, parameters: List[np.ndarray]):
    """Inject parameters into the model (only updating the trainable ones)."""
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    for param, new_val in zip(trainable_params, parameters):
        param.data = torch.tensor(new_val, dtype=param.dtype)

def train(net, trainloader, epochs, pos_weight):
    """Train the network on the training set."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Only optimize the trainable parameters
    optimizer = optim.Adam(net.get_trainable_parameters(), lr=1e-3, weight_decay=1e-4)
    
    net.train()
    for _ in range(epochs):
        for batch_x, batch_y in trainloader:
            optimizer.zero_grad()
            logits = net(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = nn.BCEWithLogitsLoss()
    net.eval()
    
    loss = 0.0
    y_true = []
    y_pred_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in testloader:
            logits = net(batch_x)
            loss += criterion(logits, batch_y).item() * batch_x.size(0)
            
            probs = torch.sigmoid(logits)
            y_true.extend(batch_y.numpy())
            y_pred_probs.extend(probs.numpy())
            
    loss /= len(testloader.dataset)
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # Calculate Custom Metrics
    ap = average_precision_score(y_true, y_pred_probs)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    valid_fpr_indices = np.where(fpr <= 0.01)[0]
    recall_at_1_fpr = tpr[valid_fpr_indices[-1]] if len(valid_fpr_indices) > 0 else 0.0
    
    metrics = {
        "ap": ap,
        "pr_auc": pr_auc,
        "recall_at_1_fpr": recall_at_1_fpr
    }
    
    return loss, metrics

class FraudClient(fl.client.NumPyClient):
    def __init__(self, cid: str, net: nn.Module, trainloader, testloader, pos_weight):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.pos_weight = pos_weight
        
    def get_parameters(self, config):
        return get_parameters(self.net)
        
    def fit(self, parameters, config):
        # Update local model with global parameters
        set_parameters(self.net, parameters)
        
        # Train locally
        epochs = config.get("local_epochs", 1)
        train(self.net, self.trainloader, epochs, self.pos_weight)
        
        # Return updated parameters to the server
        return get_parameters(self.net), len(self.trainloader.dataset), {}
        
    def evaluate(self, parameters, config):
        # Evaluate global model on local dataset
        set_parameters(self.net, parameters)
        loss, metrics = test(self.net, self.testloader)
        
        return float(loss), len(self.testloader.dataset), metrics

def get_client_fn(df, ae_weights_path, as_numpy=False):
    """
    Returns a client_fn that Flower's VirtualClientEngine can call to spawn clients on demand.
    """
    def client_fn(cid: str):
        bank_id = int(cid)
        print(f"\n[Client {bank_id}] Initializing...")
        
        # 1. Load ONLY this bank's data
        bank_df = df[df['BankID'] == bank_id].copy()
        
        # 2. Preprocess using our standard pipeline
        X_train, y_train, X_test, y_test = preprocess_bank_data(bank_df)
        
        # 3. Create PyTorch Tensors
        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
        
        # 4. Calculate pos_weight for class imbalance
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        # Fallback to 1 if there's no positive cases in a highly unbalanced split
        pos_weight = torch.tensor([num_neg / num_pos if num_pos > 0 else 1.0], dtype=torch.float32)
        
        batch_size = 128
        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False
        )
        
        # 5. Load Autoencoder & Instantiate Hybrid Model
        input_dim = X_train.shape[1]
        latent_dim = 16
        autoencoder = FraudAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        
        if os.path.exists(ae_weights_path):
            autoencoder.load_state_dict(torch.load(ae_weights_path))
            
        hybrid_model = HybridFraudModel(
            encoder=autoencoder.encoder, 
            input_dim=input_dim, 
            latent_dim=latent_dim, 
            concat_input=True
        )
        
        numpy_client = FraudClient(cid, hybrid_model, trainloader, testloader, pos_weight)
        return numpy_client if as_numpy else numpy_client.to_client()
        
    return client_fn
