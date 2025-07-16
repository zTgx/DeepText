import torch
import tiktoken
from model import DeepText, generate_text_simple

def text_to_token_ids(text, tokenizer):
    """
    Convert text to token IDs using the specified tokenizer.
    
    Args:
        text (str): The input text to tokenize.
        tokenizer: The tokenizer to use for encoding.
    
    Returns:
        list: A list of token IDs.
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs back to text using the specified tokenizer.
    
    Args:
        token_ids (torch.Tensor): The tensor of token IDs.
        tokenizer: The tokenizer to use for decoding.
    
    Returns:
        str: The decoded text.
    """
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate the loss for a batch of input and target data.
    
    Args:
        input_batch (torch.Tensor): The input batch of data.
        target_batch (torch.Tensor): The target batch of data.
        model: The model to use for prediction.
        device: The device to run the model on (CPU or GPU).
    Returns:
        torch.Tensor: The calculated loss.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss for a data loader.
    Args:
        data_loader (DataLoader): The data loader containing input and target batches.
        model: The model to use for prediction.
        device: The device to run the model on (CPU or GPU).
        num_batches (int, optional): The number of batches to process. If None, processes all batches.
    Returns:
        float: The average loss across the processed batches.
    """

    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    # Averages the loss over all batches
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    print(f"Train Loss: {train_loss}")
    print(f"Validation Loss: {val_loss}")

    model.train()
    
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
        model=model, idx=encoded,
        max_new_tokens=50, context_size=context_size
        )
    
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # Get the logits for the last token

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]

            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if idx_next == eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

    return idx
                                       