import os
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from src.version_3.tokenizer import HybridTokenizer

def build_hybrid_dataset(shard_dir="src/version_3/hybrid_shards", target_size_bytes=50 * 1024 * 1024, max_files=None):
    os.makedirs(shard_dir, exist_ok=True)
    
    try:
        ds = load_dataset("code_search_net", "python", split="train", streaming=True)
    except Exception as e:
        print(f"Dataset loading error: {e}")
        return
        
    tokenizer = HybridTokenizer()
    current_shard = 0
    current_size = 0
    buffer_bpe = []
    
    print("Starting hybrid data ingestion...")
    valid_files_processed = 0
    
    for idx, item in enumerate(ds):
        if max_files and valid_files_processed >= max_files:
            break
            
        code = item.get("content", item.get("whole_func_string", item.get("code", "")))
        if not code:
            continue
            
        bpe_ids = tokenizer.encode_text(code)
        buffer_bpe.extend(bpe_ids)
        current_size += len(bpe_ids) * 4 # 4 bytes per int32
        
        valid_files_processed += 1
        
        if current_size >= target_size_bytes:
            save_path = os.path.join(shard_dir, f"shard_{current_shard}.safetensors")
            tensor = torch.tensor(buffer_bpe, dtype=torch.int32)
            save_file({"bpe": tensor}, save_path)
            print(f"Saved {save_path}")
            
            current_shard += 1
            current_size = 0
            buffer_bpe = []

    if buffer_bpe and valid_files_processed > 0:
        save_path = os.path.join(shard_dir, f"shard_{current_shard}.safetensors")
        tensor = torch.tensor(buffer_bpe, dtype=torch.int32)
        save_file({"bpe": tensor}, save_path)
        print(f"Saved {save_path} (Final)")
        
    print(f"Hybrid ingestion complete. Processed {valid_files_processed} files.")

if __name__ == "__main__":
    build_hybrid_dataset(max_files=100)
