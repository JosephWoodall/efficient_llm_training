import torch
import torch.nn as nn
import torch.optim as optim
from src.arch.hybrid_mamba import HybridMambaMoE
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
            mask = batch["attention_mask"].to(device)
            targets = batch["answer_idx"].to(device) # 0, 1, 2, or 3
            
            logits, _ = model(ids)
            
            # Find the index of the last non-pad token for each sequence in the batch
            last_token_indices = mask.sum(dim=1) - 1
            
            # Extract logits for the last valid token
            last_logits = logits[torch.arange(logits.size(0)), last_token_indices, :]
            
            # Filter logits to only include A, B, C, D
            choice_logits = last_logits[:, choice_ids]
            preds = torch.argmax(choice_logits, dim=-1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            if i >= limit: break
    model.train()
    return (correct / total) * 100

def train():
    # Hybrid Model Hyperparameters
    d_model = 512 
    n_layers = 8  
    num_heads = 8
    attn_every = 4 # Attention every 4th layer
    batch_size = 1 
    grad_accum_steps = 4 
    lr = 3e-4 # Back to standard LR for non-hypernet
    vocab_size = 50257
    max_steps = 100000
    eval_every = 500
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridMambaMoE(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, num_heads=num_heads, attn_every=attn_every)
    model.to(device)
    
    # Check for latest checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    start_step = 0
    checkpoints = [f for f in os.listdir("checkpoints") if f.startswith("hybrid_step_")]
    if checkpoints:
        steps = [int(f.split("_")[-1].split(".")[0]) for f in checkpoints]
        latest_step = max(steps)
        checkpoint_path = f"checkpoints/hybrid_step_{latest_step}.pt"
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        start_step = latest_step + 1

    # Stream OpenWebText + Wikipedia
    train_loader = get_dataloader(tokenizer_name="gpt2", batch_size=batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Cosine Learning Rate Schedule with Warmup
    total_opt_steps = max_steps // grad_accum_steps
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_opt_steps, pct_start=0.05)
    
    # Fast-forward scheduler if resuming
    if start_step > 0:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(start_step // grad_accum_steps):
                scheduler.step()
    
    # Ignore the pad token in the loss calculation to prevent the model from collapsing
    criterion = nn.CrossEntropyLoss(ignore_index=50256)
    
    history = []

    
    print(f"Starting Hybrid-Mamba-MoE Training on {device}...")
    pbar = tqdm(total=max_steps, initial=start_step)
    
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        if i < start_step: continue
        if i >= max_steps: break
        
        ids = batch["input_ids"].to(device)
        labels = ids # Labels are same for causal LM
        
        outputs, aux_loss = model(ids[:, :-1])
        loss = criterion(outputs.reshape(-1, vocab_size), labels[:, 1:].reshape(-1))
        
        # Add aux loss to prevent expert collapse
        total_loss = loss + 0.1 * aux_loss
        
        # Scale loss for gradient accumulation
        (total_loss / grad_accum_steps).backward()
        
        if (i + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        pbar.update(1)
        pbar.set_description(f"Loss: {loss.item():.4f} | Aux: {aux_loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.1e}")
        
        if i > 0 and i % eval_every == 0:
            acc = evaluate_mmlu(model, device)
            print(f"\n" + "="*50)
            print(f"STEP {i} PERFORMANCE COMPARISON")
            print(f"="*50)
            print(f"{'Metric':<15} | {'Hybrid-Mamba':<14} | {'Claude 3 Opus':<13} | {'Gap':<8}")
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
            
            torch.save(model.state_dict(), f"checkpoints/hybrid_step_{i}.pt")

if __name__ == "__main__":
    train()