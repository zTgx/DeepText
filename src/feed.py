import torch
import torch.nn as nn

from activation import GELU
from config import GPT_CONFIG_124M

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.emb_dim = cfg["emb_dim"]

        self.layers = nn.Sequential(
            nn.Linear(self.emb_dim, 4 * self.emb_dim),
            GELU(),
            nn.Linear(4 * self.emb_dim, self.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape) # torch.Size([2, 3, 768])