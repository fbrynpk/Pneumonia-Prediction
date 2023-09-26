
"""
Contains PyTorch model code to instantiate a TinyVGG from the CNN Explainer website.
"""

import torch
from torch import nn

class TinyVGG(nn.Module):
  """
  Creates TinyVGG Architecture

  Replicates the TinyVGG Architecture from the CNN Explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/

  Args:
    input: An integer indicating number of input channels (color channels)
    hidden_units: An integer indicating the number of neurons between layers
    output: An integer indicating number of output units (class_names)
  """
  def __init__(self, input: int, hidden_units: int, output: int) -> None:
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2) # default stride value is same as kernel size
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2) # default stride value is same as kernel size
    )
    self.linear_layer = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*13*13,
                  out_features=output)
    )

  def forward(self, x):
    return self.linear_layer(self.conv_block_2(self.conv_block_1(x))) # benefits from operator fusion
