import sys
import torch
from backend.app.kyc.model import get_kyc_model
from backend.app.kyc_federation.client_app import KYCFlowerClient

def audit_payload_boundary():
    """
    Simulates a client parameter extraction and checks that raw image bytes 
    are not accidentally embedded in the parameters list.
    """
    model = get_kyc_model(pretrained=False)
    
    # We pass None for dataloaders since we only test parameter serialization
    client = KYCFlowerClient(model=model, train_loader=None, val_loader=None, device="cpu")
    
    # 1. Extract parameters to be sent to server
    params = client.get_parameters(config={})
    
    # 2. Check types
    is_numpy = all(type(p).__name__ == "ndarray" for p in params)
    print(f"All parameters are numpy arrays: {is_numpy}")
    
    # 3. Check parameter shapes to ensure they match model weights, not image sizes (e.g., 224x224x3)
    # A raw image is typically a 3D or 4D array depending on batch.
    has_image_shape = any(p.ndim >= 3 and p.shape[-2:] == (224, 224) for p in params)
    print(f"Contains tensor with image dimensions (224x224): {has_image_shape}")
    
    # 4. Measure payload size
    payload_size_mb = sum(sys.getsizeof(p) for p in params) / (1024 * 1024)
    print(f"Estimated payload size per update: {payload_size_mb:.2f} MB")
    
    assert is_numpy, "Payload contains non-array objects."
    assert not has_image_shape, "Payload appears to contain raw image tensors!"
    print("Audit passed: Payload boundary is secure at the application level.")

if __name__ == "__main__":
    audit_payload_boundary()
