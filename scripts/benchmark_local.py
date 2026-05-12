import torch
from src.arch.bit_mamba_moe import BitMambaMoE
from transformers import AutoTokenizer

def run_benchmark():
    # Load model
    d_model = 128
    vocab_size = 50257
    model = BitMambaMoE(vocab_size=vocab_size, d_model=d_model, n_layers=2, n_experts=4)
    
    # Load weights if available, otherwise use random for demo
    try:
        model.load_state_dict(torch.load("models/bit_mamba_moe_tiny.pt"))
        print("Loaded trained model.")
    except:
        print("Using uninitialized model for benchmarking.")
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Simple Benchmark Questions
    questions = [
        "What is 2 + 2?",
        "The capital of France is",
        "Explain the concept of gravity in one sentence."
    ]
    
    print("\n--- Running Local Benchmark ---")
    for q in questions:
        input_ids = tokenizer.encode(q, return_tensors="pt")
        
        with torch.no_grad():
            # Generate next 20 tokens
            output_ids = input_ids
            for _ in range(20):
                logits = model(output_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                output_ids = torch.cat([output_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Q: {q}")
        print(f"A: {response}\n")

if __name__ == "__main__":
    run_benchmark()
