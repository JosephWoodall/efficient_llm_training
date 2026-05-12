import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from src.data.mmlu_streamer import get_mmlu_loader

# --- ARCHITECTURE: Hyper-BitNet (Best Performing) ---

class WeightGenerator(nn.Module):
    def __init__(self, seed_dim, out_shape):
        super().__init__()
        self.out_shape = out_shape
        self.net = nn.Sequential(
            nn.Linear(seed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, torch.prod(torch.tensor(out_shape)).item())
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

class HyperBitModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, seed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layer = HyperBitLinear(d_model, d_model, seed_dim)
        self.output = nn.Linear(d_model, vocab_size)
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = F.silu(self.layer(x))
        return self.output(x)

# --- MMLU EVALUATION & TRAINING ---

def run_experiment():
    vocab_size = 50257
    model = HyperBitModel(vocab_size=vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Load MMLU data
    loader = get_mmlu_loader(batch_size=4)
    
    print("Running Hyper-BitNet on MMLU Benchmark...")
    total_loss = 0
    correct = 0
    total = 0
    
    # Run for 10 batches to compare
    for i, batch in enumerate(tqdm(loader)):
        ids = batch["input_ids"]
        targets = batch["target_id"]
        
        # Forward pass
        logits = model(ids)
        
        # We only care about the last non-padding token prediction
        # For MMLU, the answer is expected after the "Answer:" prompt
        last_logits = logits[:, -1, :] 
        loss = criterion(last_logits, targets)
        
        # Calculate Accuracy
        preds = torch.argmax(last_logits, dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        # Training Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if i >= 9: break
        
    avg_loss = total_loss / 10
    accuracy = (correct / total) * 100
    
    print(f"\n--- MMLU Results ---")
    print(f"Hyper-BitNet Loss: {avg_loss:.4f}")
    print(f"Hyper-BitNet Accuracy: {accuracy:.2f}%")
    print(f"Claude 3 Opus Accuracy: 86.80% (Reference)")
    print(f"--------------------\n")

if __name__ == "__main__":
    run_experiment()
