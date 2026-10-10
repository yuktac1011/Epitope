import os
import torch
import torch.nn as nn
import torch.optim as optim
from backend.app.kyc.data import get_dataloaders
from backend.app.kyc.model import get_kyc_model, freeze_backbone
from backend.app.kyc.evaluate import evaluate_model
import json

def train(model, train_loader, val_loader, epochs=2, lr=1e-3, device="cpu", save_dir="backend/app/kyc/checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    model.to(device)
    
    best_ap = 0.0
    
    print(f"Starting local training on device: {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} Average Loss: {running_loss/len(train_loader):.4f}")
        
        # Validation step
        print("Evaluating on validation set...")
        metrics, _, _ = evaluate_model(model, val_loader, device=device)
        print(f"Validation Metrics: {metrics}")
        
        # Save best model
        if metrics["ap"] > best_ap:
            best_ap = metrics["ap"]
            save_path = os.path.join(save_dir, "best_kyc_model.pth")
            torch.save(model.state_dict(), save_path)
            
            metrics_path = os.path.join(save_dir, "best_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Saved new best model with AP: {best_ap:.4f}")

    print("Training complete!")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load data (Increased training samples from 8000 to 20000 for better generalization)
    train_loader, val_loader = get_dataloaders(batch_size=32, train_samples=20000, val_samples=2000)
    
    # 2. Get and prepare model
    model = get_kyc_model(pretrained=True)
    # Start by freezing the backbone, but we'll import and unfreeze the last block to learn more complex features
    from backend.app.kyc.model import unfreeze_last_block
    model = freeze_backbone(model)
    model = unfreeze_last_block(model) # Allows the model to learn dataset-specific facial features
    
    # 3. Train (Increased from 2 epochs to 5 epochs)
    train(model, train_loader, val_loader, epochs=5, device=device)
