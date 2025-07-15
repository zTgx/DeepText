import tiktoken
import torch 
from torch import nn

# Vocabulary size
# Context length
# Embedding dimension
# Number of attention heads
# Number of layers
# Dropout rate
# Query-Key-Value bias

# Configuration for a hypothetical GPT model
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class DeepText(nn.Module):
    def __init__(self, cfg):
        pass
        
    def forward(self, in_index):
        pass

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(text1, allowed_special={"<|endoftext|>"})))
batch.append(torch.tensor(tokenizer.encode(text2, allowed_special={"<|endoftext|>"})))
batch = torch.stack(batch, dim=0)
print(batch)  # torch.Size([2, 4])

torch.manual_seed(123)
model = DeepText(cfg=None)
logits = model(batch)
print(logits.shape)  # Should match the expected output shape based on the model's architecture
print(logits)
