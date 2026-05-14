import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from src.data.streamer import get_dataloader
from src.training.logger import setup_logger

# Set up logging to the current method directory
METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
logger = setup_logger("Hyper-BitNet", METHOD_DIR)

# --- ARCHITECTURE: Hyper-BitNet ---
# A small "Seed" model generates the 1.58-bit weights for a larger virtual model.

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
        # Apply ternary quantization to the generated weight
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

# --- TRAINING ---

def run_experiment():
    vocab_size = 50257
    model = HyperBitModel(vocab_size=vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = get_dataloader(batch_size=4)
    
    logger.info("Research Idea: Hyper-BitNet (Weight Generation) Training...")
    for i, batch in enumerate(tqdm(loader)):
        ids = batch["input_ids"]
        out = model(ids[:, :-1])
        loss = criterion(out.reshape(-1, vocab_size), ids[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i >= 10: break
    logger.info(f"Finished. Final Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_experiment()
