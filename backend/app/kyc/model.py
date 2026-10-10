import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def get_kyc_model(pretrained=True, num_classes=2):
    """
    Returns a MobileNetV3-Small model for KYC deepfake detection.
    Label mapping: 0 = Fake, 1 = Real
    """
    if pretrained:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
    else:
        model = mobilenet_v3_small(weights=None)
        
    # Replace the final classification layer
    # The original model.classifier is a Sequential block.
    # The last layer is a Linear layer: model.classifier[3]
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model

def freeze_backbone(model):
    """Freezes all layers except the final classification head."""
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model

def unfreeze_last_block(model):
    """Optionally unfreeze the last feature block for fine-tuning."""
    # MobileNetV3 has a features block and a classifier block
    # Unfreeze the last few layers of the features block
    for param in model.features[-2:].parameters():
        param.requires_grad = True
    return model

if __name__ == "__main__":
    # Quick sanity check
    model = get_kyc_model()
    model = freeze_backbone(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}") # Should be [1, 2]
