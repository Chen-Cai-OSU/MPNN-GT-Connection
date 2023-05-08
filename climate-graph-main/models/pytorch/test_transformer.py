import torch
import torch.nn as nn

nhead = 7
num_encoder_layers = 2

batch_size = 20
num_nodes = 3600
history_window = 28

src = torch.randn((num_nodes, batch_size, history_window))
tgt = src.clone()

transformer_model = nn.Transformer(d_model = history_window, nhead = nhead, num_encoder_layers = num_encoder_layers)

out = transformer_model(src, tgt)
print(out.size())
print('Done')
