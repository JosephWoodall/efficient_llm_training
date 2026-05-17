import os
import torch
import itertools
from datasets import load_dataset
from safetensors.torch import save_file
from src.version_3.tokenizer import HybridTokenizer

def build_hybrid_dataset(shard_dir="src/version_3/hybrid_shards", target_size_bytes=50 * 1024 * 1024, max_files=None):
    os.makedirs(shard_dir, exist_ok=True)
    
    print("Loading code and conversational datasets...")
    try:
        ds_code = load_dataset("code_search_net", "python", split="train", streaming=True)
        ds_chat = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    except Exception as e:
        print(f"Dataset loading error: {e}")
        return
        
    tokenizer = HybridTokenizer()
    current_shard = 0
    current_size = 0
    buffer_bpe = []
    
    print("Starting hybrid data ingestion...")
    valid_files_processed = 0
    
    # Interleave the two datasets
    # alpaca format: instruction, input, output
    for idx, (code_item, chat_item) in enumerate(itertools.zip_longest(ds_code, ds_chat, fillvalue=None)):
        if max_files and valid_files_processed >= max_files:
            break
            
        # Process Code
        if code_item:
            code = code_item.get("content", code_item.get("whole_func_string", code_item.get("code", "")))
            if code:
                bpe_ids = tokenizer.encode_text(code)
                buffer_bpe.extend(bpe_ids)
                current_size += len(bpe_ids) * 4
                valid_files_processed += 1
            
        # Process Chat / Instruction
        if chat_item:
            instruction = chat_item.get("instruction", "")
            chat_input = chat_item.get("input", "")
            output = chat_item.get("output", "")
            
            if instruction and output:
                prompt = instruction
                if chat_input:
                    prompt += f"\n{chat_input}"
                
                # Format into conversational template
                chat_text = f"<|user|>\n{prompt}\n<|assistant|>\n{output}\n"
                bpe_ids_chat = tokenizer.encode_text(chat_text)
                buffer_bpe.extend(bpe_ids_chat)
                current_size += len(bpe_ids_chat) * 4
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
        
    print(f"Hybrid ingestion complete. Processed {valid_files_processed} items.")

if __name__ == "__main__":
    build_hybrid_dataset(max_files=None)
