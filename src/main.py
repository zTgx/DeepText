import torch 
import tiktoken
from model import DeepText
from utils import generate, text_to_token_ids, token_ids_to_text
from config import GPT_CONFIG_124M

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

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)


model.eval()
token_ids = generate(
model=model,
idx=text_to_token_ids("Every effort moves you", tokenizer),
max_new_tokens=15,
context_size=GPT_CONFIG_124M["context_length"],
top_k=25,
temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

