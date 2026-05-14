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
logger = setup_logger("Diff-SSM", METHOD_DIR)

# --- ARCHITECTURE: Differential SSM ---
# Enhances SSM with a "Global Memory" state that updates based on prediction surprise.

class DiffSSMBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.global_memory = nn.Parameter(torch.zeros(1, d_model)) # Highly compressed state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.surprise_gate = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: [batch, seq, d_model]
        res = x
        x = self.in_proj(x)
        x1, x2 = x.chunk(2, dim=-1)
        
        # Calculate "Surprise" to modulate memory update
        surprise = torch.sigmoid(self.surprise_gate(x1))
        
        # Update local state with global memory influence
        # This is a simplified differentiable memory update
        context = x1 * (1 - surprise) + self.global_memory * surprise
        
        out = self.out_proj(context * F.silu(x2))
        return out + res

class DiffSSMModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DiffSSMBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))

# --- TRAINING ---

def run_experiment():
    vocab_size = 50257
    model = DiffSSMModel(vocab_size=vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = get_dataloader(batch_size=4)
    
    logger.info("Research Idea: Differential SSM Training...")
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
