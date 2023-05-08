# Installation: pip install linear-attention-transformer
# Source: https://github.com/lucidrains/linear-attention-transformer

import time
import torch
from linear_attention_transformer import LinearAttentionTransformer

history_window = 28
heads = 4
num_nodes = 4096

print('Testing linear-attention Transformer')

start = time.time()
model = LinearAttentionTransformer(
    dim = history_window,
    heads = heads,
    depth = 1,
    max_seq_len = num_nodes,
    n_local_attn_heads = 1
).cuda()
end = time.time()
print('Done model creation:', end - start)

batch_size = 20

start = time.time()
x = torch.randn(batch_size, num_nodes, history_window).cuda()
out = model(x)
print(out.size())
end = time.time()
print('Done the first batch computation:', end - start)

start = time.time()
x = torch.randn(batch_size, num_nodes, history_window).cuda()
out = model(x)
print(out.size())
end = time.time()
print('Done the second batch computation:', end - start)

start = time.time()
x = torch.randn(batch_size, num_nodes, history_window).cuda()
out = model(x)
print(out.size())
end = time.time()
print('Done the third batch computation:', end - start)
