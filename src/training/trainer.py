import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.arch.bit_mamba_moe import BitMambaMoE
from src.data.streamer import get_dataloader
from tqdm import tqdm
import os

def train():
    # Hyperparameters
    d_model = 128
    n_layers = 2
    n_experts = 4
    batch_size = 8
    lr = 1e-3
    vocab_size = 50257  # GPT-2 vocab size
    
    # Initialize model
    model = BitMambaMoE(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_experts=n_experts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Data loader
    train_loader = get_dataloader(batch_size=batch_size)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    print(f"Starting training on {device}...")
    
    for i, batch in enumerate(tqdm(train_loader)):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        # Shift labels for causal LM
        outputs = model(input_ids[:, :-1])
        loss = criterion(outputs.reshape(-1, vocab_size), labels[:, 1:].reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients (important for low-bit training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}")
            
        if i >= 100: # Stop early for demo
            break
            
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bit_mamba_moe_tiny.pt")
    print("Model saved to models/bit_mamba_moe_tiny.pt")

if __name__ == "__main__":
    train()
