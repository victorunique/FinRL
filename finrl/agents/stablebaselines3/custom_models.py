import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class CNN1DFeaturesExtractor(BaseFeaturesExtractor):
    """
    1D CNN feature extractor for handling temporal sequential data.
    Input observation can be:
    1. Flattened by VecFrameStack (Batch, n_stack * n_features)
    2. Stacked (Batch, n_stack, n_features) - though SB3 VecFrameStack usually flattens likely.

    It reshapes the input to (Batch, n_features, n_stack) and applies 1D convolutions.
    
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features in the output vector
    :param n_stack: (int) Number of frames stacked (window size)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, n_stack: int = 10):
        super().__init__(observation_space, features_dim)
        
        # Determine effective shape
        # Case 1: (n_stack, n_features) or (n_features, n_stack) depending on how env is wrapped?
        # Standard SB3 VecFrameStack produces (n_stack, n_features, ...) if channels_first=False?
        # Actually SB3 VecFrameStack produces shape (n_stack * C, H, W) for images,
        # but for 1D arrays it produces (n_stack * n_features,).
        
        self.n_stack = n_stack
        
        # Check if input is already 2D (time, features) or 1D (flattened)
        if len(observation_space.shape) == 1:
            input_dim = observation_space.shape[0]
            if input_dim % n_stack != 0:
                raise ValueError(f"Observation shape {input_dim} is not divisible by n_stack {n_stack}")
            self.n_features = input_dim // n_stack
            self.flattened_input = True
        elif len(observation_space.shape) == 2:
            # Assume shape is (n_stack, n_features)
            # If (n_stack, n_features), typically n_stack is the first dim here if from a custom wrapper,
            # but usually gym boxes are (n_features,).
            # Let's assume standard (n_stack, n_features)
            # BUT verify n_stack matches
            if observation_space.shape[0] == n_stack:
                self.n_features = observation_space.shape[1]
            else:
                # Fallback/Inverse
                # If the user defines it differently
                self.n_features = observation_space.shape[0]  # or raise error
            self.flattened_input = False
        else:
            raise ValueError("Observation space must be 1D (flattened) or 2D.")

        # Components: Stack of Conv1d -> ReLU -> BatchNorm
        # Progressive channel dimensions: 32 -> ... -> 128
        
        # We start with n_features channels (transposed later to be channels)
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
            
            # Logic Change: Remove Global Average Pooling to preserve temporal info
            # nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            if self.flattened_input:
                sample_input = torch.zeros(1, self.n_stack * self.n_features)
            else:
                sample_input = torch.zeros(1, self.n_stack, self.n_features)
                
            n_flatten = self.cnn(self.reshape_input(sample_input)).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    
    def reshape_input(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (Batch, ...)
        
        if self.flattened_input:
            # (Batch, n_stack * n_features) -> (Batch, n_stack, n_features)
            x = observations.view(-1, self.n_stack, self.n_features)
        else:
            x = observations
             
        # Conv1d expects (Batch, Channels, Length)
        # We treat 'n_features' as Channels (independent indicators) and 'n_stack' as Length (Time)
        # Current x: (Batch, Time, Channels)
        # Target x: (Batch, Channels, Time)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.reshape_input(observations)
        x = self.cnn(x)
        x = self.linear(x)
        return x
