import torch

a = torch.rand(6)

a_norm = torch.softmax(a, dim=-1)
