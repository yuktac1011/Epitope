import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
try:
    from opacus import PrivacyEngine
    HAS_OPACUS = True
except ImportError:
    HAS_OPACUS = False

@pytest.mark.skipif(not HAS_OPACUS, reason="Opacus not installed")
def test_dp_clipping():
    # A test confirms clipped updates leave the client and unclipped gradients do not.
    # We will simulate a single batch update with and without DP and check gradient norms.
    
    X = torch.randn(10, 5)
    y = torch.randint(0, 2, (10, 1)).float()
    loader = DataLoader(TensorDataset(X, y), batch_size=2)
    
    net = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
    
    # 1. Non-DP Training (Unclipped)
    net_non_dp = copy.deepcopy(net)
    opt_non_dp = optim.SGD(net_non_dp.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()
    
    opt_non_dp.zero_grad()
    loss = criterion(net_non_dp(X), y)
    loss.backward()
    
    # Compute max grad norm of unclipped
    unclipped_grad_norm = max([p.grad.norm().item() for p in net_non_dp.parameters() if p.grad is not None])
    
    # 2. DP Training (Clipped)
    net_dp = copy.deepcopy(net)
    opt_dp = optim.SGD(net_dp.parameters(), lr=0.1)
    
    privacy_engine = PrivacyEngine()
    net_dp, opt_dp, dp_loader = privacy_engine.make_private(
        module=net_dp,
        optimizer=opt_dp,
        data_loader=loader,
        noise_multiplier=0.0, # no noise for pure clipping check
        max_grad_norm=0.01 # extreme clipping
    )
    
    opt_dp.zero_grad()
    for bx, by in dp_loader:
        out = net_dp(bx)
        loss = criterion(out, by)
        loss.backward()
        opt_dp.step()
        break # one step
        
    clipped_grad_norm = max([p.grad.norm().item() for p in net_dp.parameters() if p.grad is not None])
    
    # Assert clipping happened
    assert clipped_grad_norm < unclipped_grad_norm
    assert clipped_grad_norm <= 0.01 * 1.5 # Within small tolerance of clipping threshold
    
    # Check cumulative epsilon behavior
    eps1 = privacy_engine.get_epsilon(1e-5)
    
    # Take another step
    for bx, by in dp_loader:
        out = net_dp(bx)
        loss = criterion(out, by)
        loss.backward()
        opt_dp.step()
        break
        
    eps2 = privacy_engine.get_epsilon(1e-5)
    
    # Epsilon never decreases across rounds
    assert eps2 >= eps1
