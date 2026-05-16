import os
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from src.version_3.tokenizer import ASTTokenizer

def build_dataset(shard_dir="src/version_3/shards", target_size_bytes=50 * 1024 * 1024, max_files=None):
    os.makedirs(shard_dir, exist_ok=True)
    
    # Using the-stack-smol python subset for unauthenticated local execution if the-stack-v2 is gated
    try:
        ds = load_dataset("code_search_net", "python", split="train", streaming=True)
    except Exception as e:
        print(f"Dataset loading error: {e}")
        return
        
    tokenizer = ASTTokenizer()
    
    current_shard = 0
    current_size = 0
    buffer_structural = []
    
    print("Starting data ingestion...")
    valid_files_processed = 0
    
    for idx, item in enumerate(ds):
        if max_files and valid_files_processed >= max_files:
            break
            
        code = item.get("content", item.get("whole_func_string", item.get("code", "")))
        if not code:
            continue
            
        code_bytes = len(code.encode("utf8"))
        # We might lower the filter for local test to get enough files, 
        # but following spec: ignore files under 2KB or over 100KB
        if code_bytes < 2048 or code_bytes > 102400:
            continue
            
        try:
            tokens = tokenizer.encode(code)
            structural = tokens["structural"]
            buffer_structural.extend(structural)
            current_size += len(structural) * 4 # 4 bytes per int32
            
            valid_files_processed += 1
            
            if current_size >= target_size_bytes:
                save_path = os.path.join(shard_dir, f"shard_{current_shard}.safetensors")
                tensor = torch.tensor(buffer_structural, dtype=torch.int32)
                save_file({"structural": tensor}, save_path)
                print(f"Saved {save_path}")
                
                current_shard += 1
                current_size = 0
                buffer_structural = []
                
        except ValueError:
            # Skip files that fail strict parse
            continue

    if buffer_structural and valid_files_processed > 0:
        save_path = os.path.join(shard_dir, f"shard_{current_shard}.safetensors")
        tensor = torch.tensor(buffer_structural, dtype=torch.int32)
        save_file({"structural": tensor}, save_path)
        print(f"Saved {save_path} (Final)")
        
    print(f"Ingestion complete. Processed {valid_files_processed} files.")

if __name__ == "__main__":
    build_dataset(max_files=100)
