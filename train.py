import torch
import tiktoken
from utils import text_to_token_ids, token_ids_to_text, calc_loss_loader
from model import DeepText, generate_text_simple
from config import GPT_CONFIG_124M_TRAIN
from dataset import create_dataloader_v1

torch.manual_seed(123)
model = DeepText(cfg=GPT_CONFIG_124M_TRAIN)
model.eval()

# Example
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
model=model,
idx=text_to_token_ids(start_context, tokenizer),
max_new_tokens=10,
context_size=GPT_CONFIG_124M_TRAIN["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
train_data,
batch_size=2,
max_length=GPT_CONFIG_124M_TRAIN["context_length"],
stride=GPT_CONFIG_124M_TRAIN["context_length"],
drop_last=True,
shuffle=True,
num_workers=0
)

val_loader = create_dataloader_v1(
val_data,
batch_size=2,
max_length=GPT_CONFIG_124M_TRAIN["context_length"],
stride=GPT_CONFIG_124M_TRAIN["context_length"],
drop_last=False,
shuffle=False,
num_workers=0
)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)
print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print(f"Train Loss: {train_loss}")
print(f"Validation Loss: {val_loss}")