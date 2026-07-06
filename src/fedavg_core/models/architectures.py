import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    """
    A simple two-layer Multi-Layer Perceptron (MLP) mapping input features 
    to raw class logits.
    """
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        
        # First affine map: scales input dimension to hidden dimension
        self.layer1 = nn.Linear(input_size, hidden_size)
        
        # Pointwise nonlinearity
        self.relu = nn.ReLU()
        
        # Second affine map: scales hidden dimension to output classes
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        
        Args:
            x: Float tensor of shape (N, input_size)
            
        Returns:
            Float tensor of shape (N, num_classes) containing unnormalized logits.
        """
        x = self.layer1(x)
        x = self.relu(x)
        logits = self.layer2(x)
        
        return logits

def build_mlp_classifier(input_size: int, hidden_size: int, num_classes: int) -> nn.Module:
    """
    Factory function to instantiate the shared model architecture for FedAvg clients.
    """
    return MLPClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)