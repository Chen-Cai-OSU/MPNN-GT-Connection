import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import scipy
import os
import time

class ConvBlock(nn.Module):
    """ A simple convolutional block with BatchNorm and GELU activation. """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)  # a normalization layer for improved/more stable training
        self.activation = nn.GELU()  # a non-linearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class CNN(nn.Module):
    """ A simple convolutional network. """

    def __init__(self, channels_in, channels_out, channels_hidden):
        super().__init__()
        dim = channels_hidden
        # Define the convolutional layers
        self.conv1 = ConvBlock(channels_in, dim, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(dim, dim, kernel_size=3, padding=1)
        self.conv3 = ConvBlock(dim, dim // 2, kernel_size=3, padding=1)
        self.head = nn.Conv2d(dim // 2, channels_out, kernel_size=1, padding=0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h2 = h1 + h2  # Residual connection
        h3 = self.conv3(h2)
        h4 = self.head(h3)
        return h4
