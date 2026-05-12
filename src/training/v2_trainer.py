import torch
import torch.nn as nn
import torch.optim as optim
from src.arch.hyper_mamba_moe_v2 import HyperMambaMoEV2
from src.data.streamer import get_dataloader
from src.data.mmlu_streamer import get_mmlu_loader
from tqdm import tqdm
import os
import json

def evaluate_mmlu(model, device, limit=20):
    model.eval()
    loader = get_mmlu_loader(batch_size=4)
    correct = 0
    total = 0
    
    choices = ["A", "B", "C", "D"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in choices]
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            targets = batch["target_id"].to(device)
            logits = model(ids)
            last_logits = logits[:, -1, :]
            
            # Filter logits to only include A, B, C, D
            choice_logits = last_logits[:, choice_ids]
            preds = torch.argmax(choice_logits, dim=-1)
            
            # Map target token IDs to 0-3 index
            target_indices = torch.tensor([choice_ids.index(t.item()) for t in targets]).to(device)
            
            correct += (preds == target_indices).sum().item()
            total += targets.size(0)
            if i >= limit: break
    model.train()
    return (correct / total) * 100

def train():
    # v2 Hyperparameters: SCALED
    d_model = 512 
    n_layers = 6  
    seed_dim = 128
    batch_size = 1 
    grad_accum_steps = 4 
    lr = 1e-4 # Reduced for stability
    vocab_size = 50257
    max_steps = 100000
    eval_every = 500
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperMambaMoEV2(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, seed_dim=seed_dim)
    model.to(device)
    
    # Fresh start for scaled architecture
    os.makedirs("checkpoints", exist_ok=True)
    start_step = 0
    # Logic to only resume if architecture matches would be complex, 
    # so we'll just start fresh for this major version bump.
    # (Checkpoints are removed in the shell command before running)

    # Stream OpenWebText
    train_loader = get_dataloader(dataset_name="Skylion007/openwebtext", tokenizer_name="gpt2", batch_size=batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    
    print(f"Starting Hyper-Mamba-MoE v2 Training on {device}...")
    pbar = tqdm(total=max_steps, initial=start_step)
    
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        if i < start_step: continue
        if i >= max_steps: break
        
        ids = batch["input_ids"].to(device)
        labels = ids # Labels are same for causal LM
        
        outputs = model(ids[:, :-1])
        loss = criterion(outputs.reshape(-1, vocab_size), labels[:, 1:].reshape(-1))
        
        # Scale loss for gradient accumulation
        (loss / grad_accum_steps).backward()
        
        if (i + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        pbar.update(1)
        pbar.set_description(f"Loss: {loss.item():.4f}")
        
        # Don't evaluate at step 0 if resuming/starting to avoid immediate spike
        if i > 0 and i % eval_every == 0:
            acc = evaluate_mmlu(model, device)
            print(f"\n" + "="*50)
            print(f"STEP {i} PERFORMANCE COMPARISON (v2)")
            print(f"="*50)
            print(f"{'Metric':<15} | {'Hyper-Mamba-v2':<14} | {'Claude 3 Opus':<13} | {'Gap':<8}")
            print(f"-"*50)
            print(f"{'MMLU Accuracy':<15} | {acc:>13.2f}% | {'86.80%':>13} | {acc - 86.80:>7.2f}%")
            print(f"{'Training Loss':<15} | {loss.item():>13.4f} | {'N/A':>13} | {'-':>8}")
            print(f"="*50 + "\n")
            
            history.append({
                "step": i, 
                "loss": loss.item(), 
                "mmlu_acc": acc,
                "gap": acc - 86.80
            })
            
            torch.save(model.state_dict(), f"checkpoints/hyper_mamba_v2_step_{i}.pt")

if __name__ == "__main__":
    train()
