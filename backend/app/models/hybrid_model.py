import torch
import torch.nn as nn

class HybridFraudModel(nn.Module):
    def __init__(self, encoder, input_dim, latent_dim, concat_input=True):
        """
        Hybrid Fraud Classification Model that uses a frozen pre-trained encoder.
        
        Args:
            encoder (nn.Module): The pre-trained encoder module.
            input_dim (int): Number of original input features.
            latent_dim (int): Dimension of the encoder's latent space representation.
            concat_input (bool): If True, concatenates the latent vector with the original input.
        """
        super(HybridFraudModel, self).__init__()
        
        self.concat_input = concat_input
        self.encoder = encoder
        
        # Freeze the encoder weights so they don't update during training
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Determine the input dimension for the MLP classifier
        mlp_input_dim = latent_dim + input_dim if concat_input else latent_dim
        
        # Small MLP Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)  # Single logit for binary classification
        )
        
    def forward(self, x):
        # 1. Get latent representation from frozen encoder
        # Using torch.no_grad() here is a good safety measure to save memory, 
        # even though requires_grad=False already prevents parameter updates.
        with torch.no_grad():
            latent = self.encoder(x)
            
        # 2. Concatenate latent vector with original input features
        if self.concat_input:
            combined = torch.cat([latent, x], dim=1)
        else:
            combined = latent
            
        # 3. Feed combined vector into the MLP classifier
        logits = self.classifier(combined)
        return logits
        
    def get_trainable_parameters(self):
        """
        Returns an iterator over the model parameters that require gradients.
        Pass this to your optimizer (e.g., optim.Adam(model.get_trainable_parameters())).
        """
        return filter(lambda p: p.requires_grad, self.parameters())

# ==========================================
# Example Usage Block
# ==========================================
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from backend.app.models.autoencoder import FraudAutoencoder
    
    print("--- Testing Hybrid Model Structure ---")
    
    # 1. Define dimensions (Replace with your actual dataset sizes)
    input_features = 25  
    latent_size = 16
    
    # 2. Instantiate and load pre-trained autoencoder 
    autoencoder = FraudAutoencoder(input_dim=input_features, latent_dim=latent_size)
    # Example: autoencoder.load_state_dict(torch.load('best_autoencoder.pth'))
    
    # Extract just the encoder portion
    encoder_only = autoencoder.encoder
    
    # 3. Create the Hybrid Model
    hybrid_model = HybridFraudModel(
        encoder=encoder_only,
        input_dim=input_features,
        latent_dim=latent_size,
        concat_input=True
    )
    
    # 4. Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    hybrid_model = hybrid_model.to(device)
    
    # 5. Check trainable parameters
    # Notice that the frozen encoder parameters are excluded
    trainable_params_count = sum(p.numel() for p in hybrid_model.get_trainable_parameters())
    print(f"Number of trainable parameters in MLP head: {trainable_params_count}")
    
    # 6. Example Forward Pass
    # Create a dummy batch of 32 transaction samples
    dummy_input = torch.randn(32, input_features).to(device)
    
    # Set model to eval for the test pass (disables dropout, fixes layernorm)
    hybrid_model.eval()
    
    with torch.no_grad():
        output_logits = hybrid_model(dummy_input)
        
    print(f"Output shape: {output_logits.shape} (Expected: [32, 1])")
    print("Hybrid model forward pass successful!")
