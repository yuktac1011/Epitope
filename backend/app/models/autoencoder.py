import torch
import torch.nn as nn

class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16, dropout_rate=0.2):
        """
        PyTorch Autoencoder for preprocessed fraud features.
        
        Args:
            input_dim (int): Number of features in the input data.
            latent_dim (int): Size of the compressed latent representation (e.g., 8 or 16).
            dropout_rate (float): Dropout probability for regularization.
        """
        super(FraudAutoencoder, self).__init__()
        
        # Encoder: Compresses the input to a small latent vector
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, latent_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder: Reconstructs the original features from the latent vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, input_dim)
            # Note: No activation on the final layer because the input data 
            # is StandardScaled (can have negative and positive values > 1).
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def get_reconstruction_loss():
    """
    Returns the MSE reconstruction loss used for training the autoencoder.
    """
    return nn.MSELoss()
