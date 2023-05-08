import torch
import torch.nn as nn
import torch.nn.functional as F

# Installation: pip install linear-attention-transformer
# Source: https://github.com/lucidrains/linear-attention-transformer
from linear_attention_transformer import LinearAttentionTransformer

# Model
class Linear_Transformer(nn.Module):
    """ Linear Transformer. """

    def __init__(self, input_dim, hidden_dim, output_dim, max_seq_len = 4096, heads = 4, depth = 1, n_local_attn_heads = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.heads = heads
        self.depth = depth
        self.n_local_attn_heads = n_local_attn_heads

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.transformer = LinearAttentionTransformer(
            dim = self.hidden_dim,
            heads = self.heads,
            depth = self.depth,
            max_seq_len = self.max_seq_len,
            n_local_attn_heads = self.n_local_attn_heads
        )
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # Dimensions
        batch_size = x.size(0)
        num_nodes = x.size(1)
        assert x.size(2) == self.input_dim
        assert num_nodes <= self.max_seq_len

        # Pad the input with zeros
        if num_nodes < self.max_seq_len:
            zeros = torch.zeros(batch_size, self.max_seq_len - num_nodes, x.size(2))
            if x.is_cuda == True:
                zeros = zeros.cuda()
            x = torch.cat((x, zeros), dim = 1)

        # Model
        latent = torch.tanh(self.fc1(x))
        latent = self.transformer(latent)
        predict = self.fc2(latent)

        # Remove the zero columns
        predict = predict[:, : num_nodes, :]
        return predict

