import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models"""
    def __init__(self, **kwargs):
        super().__init__()
        # Store configuration
        self.config = kwargs
        self._build_model()
    
    @abstractmethod
    def _build_model(self):
        """Implement the model architecture"""
        pass
    
    @abstractmethod
    def forward(self, x):
        """Implement the forward pass"""
        pass
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        """Load model weights"""
        self.load_state_dict(torch.load(path))

class Generator(BaseModel):
    def __init__(self, latent_dim=128, output_shape=None, **kwargs):
        """
        Args:
            latent_dim (int): Dimension of the latent space
            output_shape (tuple): Expected shape of the output (C, H, W)
            **kwargs: Additional architecture-specific parameters
        """
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        super().__init__(**kwargs)
    
    def _build_model(self):
        """
        Implement the generator architecture
        Override this method for specific implementations
        """
        self.main = nn.ModuleList([
            # Add your layers here
        ])
    
    def forward(self, z):
        """
        Args:
            z (torch.Tensor): Latent vectors [B, latent_dim]
        Returns:
            torch.Tensor: Generated outputs
        """
        x = z
        for layer in self.main:
            x = layer(x)
        return x

class Discriminator(BaseModel):
    def __init__(self, input_shape, **kwargs):
        """
        Args:
            input_shape (tuple): Expected shape of inputs (C, H, W)
            **kwargs: Additional architecture-specific parameters
        """
        self.input_shape = input_shape
        super().__init__(**kwargs)
    
    def _build_model(self):
        """
        Implement the discriminator architecture
        Override this method for specific implementations
        """
        self.main = nn.ModuleList([
            # Add your layers here
        ])
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input data
        Returns:
            torch.Tensor: Discrimination results
        """
        for layer in self.main:
            x = layer(x)
        return x

class CustomLoss(nn.Module):
    """Template for custom loss functions"""
    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs
    
    def forward(self, predictions, targets):
        """
        Implement the loss computation
        Args:
            predictions: Model outputs
            targets: Ground truth
        Returns:
            torch.Tensor: Computed loss
        """
        pass

def create_model_instance(model_type, config):
    """Factory function to create model instances"""
    model_map = {
        'generator': Generator,
        'discriminator': Discriminator,
        # Add more model types as needed
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_map[model_type](**config)

if __name__ == "__main__":
    # Example usage
    config = {
        'generator': {
            'latent_dim': 128,
            'output_shape': (3, 64, 64),  # Example for RGB images
        },
        'discriminator': {
            'input_shape': (3, 64, 64),
        }
    }
    
    # Create models
    G = create_model_instance('generator', config['generator'])
    D = create_model_instance('discriminator', config['discriminator'])
    
    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, config['generator']['latent_dim'])
    fake_data = G(z)
    d_output = D(fake_data)
    
    # Print model info
    print(f"\nGenerator parameters: {G.count_parameters():,}")
    print(f"Discriminator parameters: {D.count_parameters():,}")
    print(f"\nGenerator output shape: {fake_data.shape}")
    print(f"Discriminator output shape: {d_output.shape}")
