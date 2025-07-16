import torch 
from torch import nn
import tiktoken

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

# DeepText model definition
# A simplified version of a GPT-like model using PyTorch’s neural network module (nn.Module).
class DeepText(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg if cfg is not None else GPT_CONFIG_124M
        
        self.vocab_size = self.cfg["vocab_size"]
        self.emb_dim = self.cfg["emb_dim"]
        self.tok_emb = nn.Embedding(self.vocab_size, self.emb_dim)

        self.context_length = self.cfg["context_length"]
        self.pos_emb = nn.Embedding(self.context_length, self.emb_dim)

        self.drop = self.cfg["drop_rate"]
        self.drop_emb = nn.Dropout(self.drop)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(self.cfg) for _ in range(self.cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(self.emb_dim, eps=1e-5)

        self.out_head = nn.Linear(self.emb_dim, self.vocab_size, bias=self.cfg["qkv_bias"])


    # Describes the data flow through the model
    def forward(self, in_index):
        batch_size, seq_len = in_index.shape
        assert seq_len <= self.context_length, "Input sequence length exceeds context length."

        tok_emb = self.tok_emb(in_index)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=in_index.device))

        x = tok_emb + pos_emb

        # Applies dropout
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        # Logits shape: (batch_size, seq_len, vocab_size)
        #  每一个 单词计算一个分数（称为 "logit"）
        return logits

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

text1_tokens = tokenizer.encode(text1, allowed_special={"<|endoftext|>"})
text2_tokens = tokenizer.encode(text2, allowed_special={"<|endoftext|>"})
print(f"Text 1 tokens: {text1_tokens}")
print(f"Text 2 tokens: {text2_tokens}")

text1_tokens_tensor = torch.tensor(text1_tokens)
text2_tokens_tensor = torch.tensor(text2_tokens)
print(f"Text 1 tokens tensor: {text1_tokens_tensor}")
print(f"Text 2 tokens tensor: {text2_tokens_tensor}")

batch.append(text1_tokens_tensor)
batch.append(text2_tokens_tensor)

# dim=0 means the new dimension will be the very first one.
batch = torch.stack(batch, dim=0)
print(batch)  # torch.Size([2, 4])

torch.manual_seed(123)

# Initialize the model and run a forward pass
model = DeepText(cfg=None)
logits = model(batch)
print(logits.shape)  # Should match the expected output shape based on the model's architecture
print(logits)
