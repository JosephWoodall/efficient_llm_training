import tree_sitter
import tree_sitter_python
from transformers import GPT2Tokenizer

class HybridTokenizer:
    def __init__(self):
        self.bpe = GPT2Tokenizer.from_pretrained("gpt2")
        self.bpe.pad_token = self.bpe.eos_token
        
        try:
            self.language = tree_sitter.Language(tree_sitter_python.language())
        except TypeError:
            self.language = tree_sitter_python.language()
            
        self.parser = tree_sitter.Parser()
        self.parser.language = self.language
        
        self.syntax_vocab = {
            "UNK": 0, "PAD": 1, "module": 2, "expression_statement": 3,
            "assignment": 4, "identifier": 5, "integer": 6,
            "function_definition": 7, "def": 8, "parameters": 9,
            "block": 10, "return_statement": 11, "return": 12,
            "binary_operator": 13, "if_statement": 14, "if": 15,
            "string": 16, "call": 17, "TEXT_MODE": 18
        }
        self.id_to_syntax = {v: k for k, v in self.syntax_vocab.items()}
        
        self.vocab_size = self.bpe.vocab_size
        self.syntax_vocab_size = len(self.syntax_vocab)
        
    def encode_text(self, text: str):
        # Truncate to max sequence length to prevent GPT2Tokenizer errors
        return self.bpe.encode(text, truncation=True, max_length=1024)
        
    def decode_text(self, ids):
        return self.bpe.decode(ids)
        
    def get_syntax_state(self, code_str: str):
        """Returns the current structural state of the AST."""
        if not code_str.strip():
            return "module"
            
        tree = self.parser.parse(bytes(code_str, "utf8"))
        if tree.root_node.has_error:
            return "UNK" # Indicates invalid syntax
            
        # Find the deepest right-most node to determine current state
        cursor = tree.walk()
        while cursor.goto_last_child():
            pass
        return cursor.node.type

if __name__ == "__main__":
    tk = HybridTokenizer()
    print("BPE encoded:", tk.encode_text("def hello():\n    pass"))
    print("AST State:", tk.get_syntax_state("def hello():\n"))
