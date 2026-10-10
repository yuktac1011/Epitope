import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from flwr.client import NumPyClient

from backend.app.kyc.data import get_dataloaders
from backend.app.kyc.evaluate import evaluate_model

def get_partition_dataloaders(node_id: int, num_partitions: int):
    # Total 1500 samples partitioned across banks (500 per bank) for fast CPU simulation
    total_samples = 1500
    samples_per_partition = total_samples // num_partitions

    train_loader, val_loader = get_dataloaders(batch_size=32, train_samples=total_samples, val_samples=150)
    train_dataset = train_loader.dataset

    start_idx = (node_id % num_partitions) * samples_per_partition
    end_idx = start_idx + samples_per_partition

    subset = torch.utils.data.Subset(train_dataset, range(start_idx, end_idx))
    partition_loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)

    return partition_loader, val_loader

class KYCFlowerClient(NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()

        epochs = config.get("local_epochs", 1)
        lr = config.get("lr", 1e-3)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        for _ in range(epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics, _, _ = evaluate_model(self.model, self.val_loader, device=self.device)
        loss = 0.0
        return loss, len(self.val_loader.dataset), {"ap": float(metrics["ap"]), "fpr": float(metrics["fpr"])}
