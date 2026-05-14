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
logger = setup_logger("Shared-Rec-MoE", METHOD_DIR)

# --- ARCHITECTURE: Shared-Recurrent MoE ---
# Combines a persistent "Working Memory" expert with specialized "Knowledge" experts.

class SharedRecMoEBlock(nn.Module):
    def __init__(self, d_model, n_experts=4):
        super().__init__()
        self.shared_expert = nn.Linear(d_model, d_model) # Persistent state
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts)
        
    def forward(self, x):
        # Persistent working memory path
        shared_out = F.silu(self.shared_expert(x))
        
        # Specialized knowledge path
        logits = self.gate(x)
        expert_idx = torch.argmax(logits, dim=-1)
        
        expert_results = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)
            if mask.any():
                expert_results[mask] = expert(x[mask])
        
        return x + shared_out + expert_results

class SharedRecMoEModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([SharedRecMoEBlock(d_model) for _ in range(n_layers)])
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

# --- TRAINING ---

def run_experiment():
    vocab_size = 50257
    model = SharedRecMoEModel(vocab_size=vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = get_dataloader(batch_size=4)
    
    logger.info("Research Idea: Shared-Recurrent MoE Training...")
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
