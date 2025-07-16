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

