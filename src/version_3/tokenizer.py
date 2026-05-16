import tree_sitter
import tree_sitter_python

class ASTTokenizer:
    def __init__(self):
        try:
            self.language = tree_sitter.Language(tree_sitter_python.language())
        except TypeError:
            self.language = tree_sitter_python.language()
        
        self.parser = tree_sitter.Parser()
        self.parser.language = self.language
        
        self.syntax_vocab = {
            "UNK": 0,
            "PAD": 1,
            "module": 2,
            "expression_statement": 3,
            "assignment": 4,
            "identifier": 5,
            "integer": 6,
            "function_definition": 7,
            "def": 8,
            "parameters": 9,
            "block": 10,
            "return_statement": 11,
            "return": 12,
            "binary_operator": 13,
            "if_statement": 14,
            "if": 15,
            "string": 16,
            "call": 17,
        }
        self.id_to_syntax = {v: k for k, v in self.syntax_vocab.items()}
        
    def encode(self, source_code: str):
        tree = self.parser.parse(bytes(source_code, "utf8"))
        if tree.root_node.has_error:
            raise ValueError("Corrupted or invalid syntax sequence: Parse error in source code.")
            
        structural_stream = []
        literal_stream = []
        
        def traverse(node):
            node_type = node.type
            structural_stream.append(self.syntax_vocab.get(node_type, self.syntax_vocab["UNK"]))
            
            if node.is_named and not node.children:
                literal_stream.append(node.text.decode("utf8"))
            else:
                for child in node.children:
                    traverse(child)
                    
        traverse(tree.root_node)
        return {"structural": structural_stream, "literal": literal_stream}

if __name__ == "__main__":
    tk = ASTTokenizer()
    code = "def add(a, b):\n    return a + b"
    print("Encoded:", tk.encode(code))
