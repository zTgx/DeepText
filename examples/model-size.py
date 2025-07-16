from transformers import pipeline
generation_gpt2 = pipeline("text-generation", model="openai-community/gpt2")

def model_size(model):
    return sum(t.numel() for t in model.parameters())

if __name__ == "__main__":
    print("Model size in parameters:")
    print(f"GPT2 size: {model_size(generation_gpt2.model)/1000**2:.1f}M parameters")