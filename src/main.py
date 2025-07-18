import torch 
import tiktoken
from model import DeepText
from utils import generate, text_to_token_ids, token_ids_to_text
from config import GPT_CONFIG_124M

def setup_tokenizer():
    """Initialize and return the tokenizer"""
    return tiktoken.get_encoding("gpt2")

def prepare_batch(texts, tokenizer):
    """Prepare a batch of tokenized texts"""
    batch = []
    for i, text in enumerate(texts, 1):
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        print(f"Text {i} tokens: {tokens}")
        
        tokens_tensor = torch.tensor(tokens)
        print(f"Text {i} tokens tensor: {tokens_tensor}")
        
        batch.append(tokens_tensor)
    
    # Stack along the first dimension to create a batch
    return torch.stack(batch, dim=0)

def run_model_forward_pass(model, batch):
    """Run a forward pass through the model"""
    logits = model(batch)
    print("Logits shape:", logits.shape)
    print("Logits:", logits)
    return logits

def generate_sample_text(model, tokenizer, config):
    """Generate sample text using the model"""
    start_context = "Every effort moves you"
    print("\nGenerating text continuation...")
    
    model.eval()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=15,
        context_size=config["context_length"],
        top_k=25,
        temperature=1.4
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

def main():
    # Initialize components
    torch.manual_seed(123)
    tokenizer = setup_tokenizer()
    model = DeepText(cfg=None)
    
    # Prepare and process batch
    texts = ["Every effort moves you", "Every day holds a"]
    batch = prepare_batch(texts, tokenizer)
    print("Batch tensor:", batch)
    print("Batch shape:", batch.shape)
    
    # Run model forward pass
    run_model_forward_pass(model, batch)
    
    # Generate sample text
    generate_sample_text(model, tokenizer, GPT_CONFIG_124M)

if __name__ == "__main__":
    main()