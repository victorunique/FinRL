import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class CNN1DFeaturesExtractor(BaseFeaturesExtractor):
    """
    1D CNN feature extractor for handling temporal sequential data.
    Input observation is expected to be flattened by VecFrameStack with shape (n_stack * n_features).
    It reshapes the input to (Batch, n_features, n_stack) and applies 1D convolutions.
    
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features in the output vector
    :param n_stack: (int) Number of frames stacked (window size)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, n_stack: int = 10):
        # The features_dim argument matches the number of neurons in the last layer
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        self.n_stack = n_stack
        self.n_features = input_dim // n_stack
        
        # Verify that the input shape is divisible by n_stack
        if input_dim % n_stack != 0:
            raise ValueError(f"Observation shape {input_dim} is not divisible by n_stack {n_stack}")

        # Components: Stack of Conv1d -> ReLU -> BatchNorm
        # Progressive channel dimensions: 32 -> ... -> 128
        
        # We start with n_features channels
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=self.n_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            # Layer 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            # Layer 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            # Create a dummy input of shape (1, input_dim)
            sample_input = torch.zeros(1, input_dim)  # (Batch, Flattened_Input)
            # Reshape logic matches forward()
            reshaped = sample_input.view(1, self.n_stack, self.n_features).permute(0, 2, 1)  # (B, F, T)
            n_flatten = self.cnn(reshaped).shape[1]

        # If the output of CNN doesn't match features_dim, we could add a linear layer
        # But the design says "terminated by Global Average Pooling... fixed-length feature representation"
        # Since the last conv has 128 channels and we do AdaptiveAvgPool1d(1), the output is 128.
        # So it should match features_dim=128.
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (Batch, n_stack * n_features)
        # Reshape to (Batch, n_stack, n_features)
        x = observations.view(-1, self.n_stack, self.n_features)
        # Permute to (Batch, n_features, n_stack) for Conv1d
        x = x.permute(0, 2, 1)

        return self.cnn(x)
