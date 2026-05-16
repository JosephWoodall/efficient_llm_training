import os
import torch
from src.version_3.model import TSSM
from src.version_3.tokenizer import ASTTokenizer

def main():
    print("=== TSSM Interactive CLI ===")
    
    tokenizer = ASTTokenizer()
    # Initialize the model with the exact same architecture params used in training
    model = TSSM(vocab_size=len(tokenizer.syntax_vocab), d_model=128, n_layers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    checkpoint_path = "src/version_3/checkpoints/tssm_local_prod.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded trained checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: No trained checkpoint found at {checkpoint_path}. Using untrained weights.")
        
    model.eval()
    
    print("\n[System]: TSSM operates purely on Abstract Syntax Trees (AST).")
    print("[System]: It predicts structural logic, not raw text.")
    print("[System]: Please provide a valid Python snippet (e.g., 'def calculate(a, b):') to seed the state space.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input(">>> ")
            if user_input.strip().lower() == "exit":
                break
                
            if not user_input.strip():
                continue
                
            # Attempt to parse the user's Python code
            tokens = tokenizer.encode(user_input)
            structural_ids = tokens["structural"]
            
            input_structure = [tokenizer.id_to_syntax.get(tid, "UNK") for tid in structural_ids]
            print(f"\n[Parsed Input AST]: {' -> '.join(input_structure)}")
            
            input_tensor = torch.tensor([structural_ids], device=device).long()
            
            print("[Generating...]")
            with torch.no_grad():
                generated_tensor = model.generate(input_tensor, max_length=15)
                
            generated_ids = generated_tensor[0].cpu().tolist()
            
            # Filter to show only the newly predicted syntax tokens
            new_ids = generated_ids[len(structural_ids):]
            generated_structure = [tokenizer.id_to_syntax.get(tid, "UNK") for tid in new_ids]
            
            print(f"[Predicted Output AST]: {' -> '.join(generated_structure)}\n")
            
        except ValueError as e:
            print(f"\n[Syntax Error]: {e}")
            print("Remember to enter structurally valid Python code for the model to parse.\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Unexpected Error]: {e}\n")

if __name__ == "__main__":
    main()
