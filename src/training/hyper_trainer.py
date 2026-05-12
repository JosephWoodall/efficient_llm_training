import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.arch.bit_mamba_moe import weight_quant # Reuse quantization logic
from src.data.streamer import get_dataloader
from src.data.mmlu_streamer import get_mmlu_loader
from tqdm import tqdm
import os
import json

# --- ARCHITECTURE RE-DEFINITION (Self-contained for the trainer) ---

class WeightGenerator(nn.Module):
    def __init__(self, seed_dim, out_shape):
        super().__init__()
        self.out_shape = out_shape
        self.net = nn.Sequential(
            nn.Linear(seed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, torch.prod(torch.tensor(out_shape)).item())
        )
    def forward(self, seed):
        w = self.net(seed).view(self.out_shape)
        scale = w.abs().mean() + 1e-7
        w_quant = torch.round(w / scale).clamp(-1, 1)
        return w + (w_quant - w).detach()

class HyperBitLinear(nn.Module):
    def __init__(self, in_features, out_features, seed_dim):
        super().__init__()
        self.generator = WeightGenerator(seed_dim, (out_features, in_features))
        self.seed = nn.Parameter(torch.randn(seed_dim))
        self.bias = nn.Parameter(torch.zeros(out_features))
    def forward(self, x):
        w = self.generator(self.seed)
        return F.linear(x, w, self.bias)

# Note: Using functional F for consistency
import torch.nn.functional as F

class HyperBitModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, seed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "linear1": HyperBitLinear(d_model, d_model * 4, seed_dim),
                "linear2": HyperBitLinear(d_model * 4, d_model, seed_dim)
            }) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            res = x
            x = F.silu(layer["linear1"](x))
            x = layer["linear2"](x)
            x = x + res
        return self.output(self.norm(x))

# --- TRAINING LOGIC ---

def evaluate_mmlu(model, device, limit=20):
    model.eval()
    loader = get_mmlu_loader(batch_size=4)
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            targets = batch["target_id"].to(device)
            logits = model(ids)
            last_logits = logits[:, -1, :]
            preds = torch.argmax(last_logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            if i >= limit: break
    model.train()
    return (correct / total) * 100

def train():
    # Hyperparameters for intensive run
    d_model = 128
    n_layers = 4
    seed_dim = 64
    batch_size = 4
    lr = 5e-4
    vocab_size = 50257
    max_steps = 100000  # Massive scaling
    eval_every = 500   # Less frequent eval to prioritize throughput
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperBitModel(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, seed_dim=seed_dim)
    model.to(device)
    
    # Use OpenWebText for massive-scale streaming (no disk storage of data)
    train_loader = get_dataloader(dataset_name="Skylion007/openwebtext", tokenizer_name="gpt2", batch_size=batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # SAVE PROGRESS: Re-enabling checkpointing as only the DATA should be streamed to save disk.
    os.makedirs("checkpoints", exist_ok=True)
    start_step = 0
    checkpoints = [f for f in os.listdir("checkpoints") if f.startswith("hyper_bitnet_streaming_step_")]
    if checkpoints:
        steps = [int(f.split("_")[-1].split(".")[0]) for f in checkpoints]
        latest_step = max(steps)
        checkpoint_path = f"checkpoints/hyper_bitnet_streaming_step_{latest_step}.pt"
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        start_step = latest_step + 1

    history = []
    
    print(f"Starting Streaming Data Training on {device} (Minimal Disk Footprint)...")
    pbar = tqdm(total=max_steps, initial=start_step)
    
    for i, batch in enumerate(train_loader):
        if i < start_step: continue
        if i >= max_steps: break
        
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(ids[:, :-1])
        loss = criterion(outputs.reshape(-1, vocab_size), labels[:, 1:].reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        pbar.update(1)
        pbar.set_description(f"Loss: {loss.item():.4f}")
        
        if i % eval_every == 0:
            acc = evaluate_mmlu(model, device)
            print(f"\n" + "="*50)
            print(f"STEP {i} PERFORMANCE COMPARISON")
            print(f"="*50)
            print(f"{'Metric':<15} | {'Hyper-BitNet':<12} | {'Claude 3 Opus':<13} | {'Gap':<8}")
            print(f"-"*50)
            print(f"{'MMLU Accuracy':<15} | {acc:>11.2f}% | {'86.80%':>13} | {acc - 86.80:>7.2f}%")
            print(f"{'Training Loss':<15} | {loss.item():>11.4f} | {'N/A':>13} | {'-':>8}")
            print(f"="*50 + "\n")
            
            history.append({
                "step": i, 
                "loss": loss.item(), 
                "mmlu_acc": acc,
                "opus_target": 86.80,
                "gap": acc - 86.80
            })
            
            # Save checkpoint (only ~300MB)
            torch.save(model.state_dict(), f"checkpoints/hyper_bitnet_streaming_step_{i}.pt")
            
    # Final Save
    torch.save(model.state_dict(), "models/hyper_bitnet_streaming_final.pt")
    with open("training_history_streaming.json", "w") as f:
        json.dump(history, f)
        
    print("Training Complete. Model preserved on disk (minimal size).")

if __name__ == "__main__":
    train()
