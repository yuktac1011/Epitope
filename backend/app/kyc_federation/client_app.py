import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from collections import OrderedDict
from backend.app.kyc.model import get_kyc_model, freeze_backbone
from backend.app.kyc.data import get_dataloaders
from backend.app.kyc.evaluate import evaluate_model
from flwr.client import NumPyClient, ClientApp

# Define a function to partition data based on node ID (simulated clients)
def get_client_dataloader(node_id: int, num_partitions: int):
    # For simulation, divide 3000 samples across partitions (1000 per simulated bank)
    total_samples = 3000 
    samples_per_partition = total_samples // num_partitions
    
    train_loader, val_loader = get_dataloaders(batch_size=32, train_samples=total_samples, val_samples=300)
    
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
        
        # Local training configuration
        epochs = config.get("local_epochs", 1)
        lr = config.get("lr", 1e-3)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        for epoch in range(epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Measure payload size roughly (for auditing)
        import sys
        payload_bytes = sum(sys.getsizeof(p) for p in parameters)
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"payload_bytes": payload_bytes}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics, _, _ = evaluate_model(self.model, self.val_loader, device=self.device)
        loss = 0.0 # Standardize loss if required
        return loss, len(self.val_loader.dataset), {"ap": metrics["ap"], "fpr": metrics["fpr"]}

def client_fn(context: fl.common.Context) -> fl.client.Client:
    num_partitions = context.run_config.get("num-partitions", context.run_config.get("num_partitions", 3))
    raw_node = context.node_config.get("partition-id", getattr(context, "node_id", 0))
    try:
        node_id = int(raw_node) % num_partitions
    except Exception:
        node_id = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load partitioned data
    train_loader, val_loader = get_client_dataloader(node_id, num_partitions)
    
    # Initialize model
    model = get_kyc_model(pretrained=True)
    model = freeze_backbone(model)
    
    return KYCFlowerClient(model, train_loader, val_loader, device).to_client()

# Create the ClientApp
app = ClientApp(
    client_fn=client_fn,
)
