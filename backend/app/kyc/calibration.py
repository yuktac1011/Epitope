import torch
import torch.nn as nn
import numpy as np

class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to logits:
    p = softmax(logits / T)
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            scaled = self.forward(logits)
            return torch.softmax(scaled, dim=-1)

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 100):
        """
        Fits temperature parameter on validation set to minimize NLL.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())
