import torch

def build_Adam(params,lr,weight_decay):
    return torch.optim.AdamW(params=params,lr=lr, betas=(0.9, 0.999),eps=1e-6, weight_decay=weight_decay)

