import torch
from src.arch.hybrid_mamba import HybridMambaMoE
from src.training.elf_eval import ELFEvaluator
from transformers import AutoTokenizer
import argparse

def run_elf_benchmark(model_path=None, num_samples=10, device="cpu"):
    # Initialize Model
    vocab_size = 50257
    d_model = 256
    model = HybridMambaMoE(vocab_size=vocab_size, d_model=d_model, n_layers=4)
    
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}. Using random initialization.")
            
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # ELF Evaluator
    evaluator = ELFEvaluator(device=device)
    
    # Prompts for evaluation (Subset of OpenWebText-like or Alpaca-like prompts)
    prompts = [
        "The future of artificial intelligence is",
        "The capital of France is",
        "In a hole in the ground there lived a",
        "Quantum physics is a branch of science that",
        "The best way to learn programming is"
    ]
    
    print("\n--- Starting ELF Benchmark Flow ---")
    results = evaluator.evaluate(
        model, 
        tokenizer, 
        prompts, 
        num_samples=num_samples, 
        max_new_tokens=30
    )
    
    print("\n--- ELF Benchmark Results ---")
    print(f"Generative Perplexity (Gen. PPL): {results['gen_ppl']:.4f}")
    print(f"Unigram Entropy: {results['unigram_entropy']:.4f}")
    print(f"Samples Evaluated: {results['num_samples']}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ELF Benchmark Flow")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model weights")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")
    
    args = parser.parse_args()
    run_elf_benchmark(model_path=args.model_path, num_samples=args.num_samples, device=args.device)
