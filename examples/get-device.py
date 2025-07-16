import torch

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    device = get_device()
    print(f"On device: {device}")