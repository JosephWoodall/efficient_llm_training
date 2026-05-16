import os
import torch
from src.version_3.model import HybridTSSM
from src.version_3.tokenizer import HybridTokenizer

def main():
    print("=== Hybrid TSSM Interactive CLI ===")
    
    tokenizer = HybridTokenizer()
    model = HybridTSSM(vocab_size=tokenizer.vocab_size, syntax_vocab_size=tokenizer.syntax_vocab_size, d_model=1024, n_layers=12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    checkpoint_path = "src/version_3/checkpoints/hybrid_tssm_local.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded trained hybrid checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: No trained checkpoint found at {checkpoint_path}. Run pipeline & train first.")
        
    model.eval()
    
    print("\n[System]: Hybrid Mode Active.")
    print("[System]: You can converse normally in English.")
    print("[System]: If the model outputs ```python, the AST Dynamic Gate strictly constrains output to valid code.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input(">>> ")
            if user_input.strip().lower() == "exit":
                break
                
            if not user_input.strip():
                continue
                
            # Parse user input as BPE
            bpe_ids = tokenizer.encode_text(user_input)
            input_tensor = torch.tensor([bpe_ids], device=device).long()
            
            with torch.no_grad():
                generated_tensor = model.generate(input_tensor, max_length=20, tokenizer=tokenizer)
                
            generated_ids = generated_tensor[0].cpu().tolist()
            new_ids = generated_ids[len(bpe_ids):]
            
            output_text = tokenizer.decode_text(new_ids)
            print(f"\n[Model]: {output_text}\n")
            
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Unexpected Error]: {e}\n")

if __name__ == "__main__":
    main()
