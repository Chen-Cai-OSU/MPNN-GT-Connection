import torch
import torch.nn as nn
import torch.nn.functional as F

# Model
class MLP(nn.Module):
    """ A simple Multilayer Perceptron. """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))

