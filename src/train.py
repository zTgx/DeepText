import torch
import tiktoken
from utils import text_to_token_ids, token_ids_to_text, calc_loss_loader, calc_loss_batch, generate_and_print_sample, evaluate_model, generate
from model import DeepText, generate_text_simple
from config import GPT_CONFIG_124M_TRAIN
from dataset import create_dataloader_v1

# torch.manual_seed(123)
# model = DeepText(cfg=GPT_CONFIG_124M_TRAIN)
# model.eval()

# # Example
# start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
# token_ids = generate_text_simple(
# model=model,
# idx=text_to_token_ids(start_context, tokenizer),
# max_new_tokens=10,
# context_size=GPT_CONFIG_124M_TRAIN["context_length"]
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

file_path = "../data/the-verdict.txt"
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

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# model.to(device)

# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device)
#     val_loss = calc_loss_loader(val_loader, model, device)

# print(f"Train Loss: {train_loss}")
# print(f"Validation Loss: {val_loss}")

def train_model_simple(model, train_loader, val_loader,
optimizer, device, num_epochs,
eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                        f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(model, tokenizer, device, start_context )

    return train_losses, val_losses, track_tokens_seen


torch.manual_seed(123)
model = DeepText(GPT_CONFIG_124M_TRAIN)
model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(),lr=0.0004, weight_decay=0.1)
# num_epochs = 5
# train_losses, val_losses, tokens_seen = train_model_simple(
# model, train_loader, val_loader, optimizer, device,
# num_epochs=num_epochs, eval_freq=5, eval_iter=5,
# start_context="Every effort moves you", tokenizer=tokenizer
# )

# # Plot
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
#     fig, ax1 = plt.subplots(figsize=(5, 3))
#     ax1.plot(epochs_seen, train_losses, label="Training loss")
#     ax1.plot(
#     epochs_seen, val_losses, linestyle="-.", label="Validation loss"
#     )
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend(loc="upper right")
#     ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
#     ax2 = ax1.twiny()
#     ax2.plot(tokens_seen, train_losses, alpha=0)
#     ax2.set_xlabel("Tokens seen")
#     fig.tight_layout()
#     plt.show()

# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


torch.save(model.state_dict(), "model.pth")

# torch.manual_seed(123)
# token_ids = generate(
# model=model,
# idx=text_to_token_ids("Every effort moves you", tokenizer),
# max_new_tokens=15,
# context_size=GPT_CONFIG_124M_TRAIN["context_length"],
# top_k=25,
# temperature=1.4
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))