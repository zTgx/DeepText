import urllib.request

url = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")

file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Length of text: {len(raw_text)} characters")
print(raw_text[:99])

import re
# text = "Hello, workd. This , is a test."
# result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# print(result)

# result = [item.strip() for item in result if item.strip()]
# print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

all_tokens = sorted(set(preprocessed))

# two special tokens, <unk> and <|endoftext|>
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab_size = len(all_tokens)
print(f"Total words: {vocab_size}")

vocab = {word: idx for idx, word in enumerate(all_tokens)}
for i, item in enumerate(vocab.items()):
    print(f"{i}: {item}")
    if  i >= 50:
        break
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

class SimpleTokennizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {idx: word for word, idx in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # replace the specials tokens
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        return [self.str_to_int[word] for word in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[idx] for idx in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
tokenizer = SimpleTokennizer(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

from importlib.metadata import version
import tiktoken
print("tiktoken version: ", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")
text = (
"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
print(tokenizer.decode(integers))

#Tokenize the whole “The Verdict” short story using the BPE tokenizer
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# remove the first 50 tokens
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1: context_size+1]
print(x, y)
print(tokenizer.decode(x), tokenizer.decode(y))

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)


import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

# __len__ 和 __getitem__ 是 Python 中的​​特殊方法（魔术方法）​​，用于实现自定义容器的基本行为
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                        stride=128, shuffle=True, drop_last=True,
                        num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,  # drops the last batch if it is shorter than the
        num_workers=num_workers  # specified batch_size to prevent loss spikes during training
    )
    return dataloader

vocab_size = 50257
output_dim = 256
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

print(embedding_layer(torch.tensor([3])))

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = embedding_layer(inputs)
print(token_embeddings.shape)

### Attention mechanism
# Embeddings tensoer
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] # step
)
print(inputs.shape)

# Query token: Token index=1 as Query token inputs[1]
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(query, x_i)

# Attention Score: embedding elements [dot product] of the query token with all input tokens
print(attn_scores_2)

# This is a more efficient way to compute the dot-product attention scores
# for a single query against all inputs.
attn_scores_2 = torch.matmul(inputs, query)

# attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())

# Attention weights: Softmax normalization: Convert attention scores to Attention weights.
# 每个元素 σ(z) 是一个介于 0 和 1 之间的概率值。
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# Context vector: Weighted sum of the input embeddings using the attention weights.
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

# tensor([0.4419, 0.6515, 0.5683])
print(context_vec_2)

# The for-loop can be replaced by a more efficient matrix-vector multiplication.
context_vec_2 = torch.matmul(attn_weights_2, inputs)
print(context_vec_2)


# All inputs
attn_scores = inputs @ inputs.T
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)


