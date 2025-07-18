import torch 
from torch import nn
import tiktoken

from norm import LayerNorm
from config import GPT_CONFIG_124M
from block import TransformerBlock

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

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]

            probas = torch.softmax(logits, dim=-1)

            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)

    return idx