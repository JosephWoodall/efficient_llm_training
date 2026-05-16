import torch
import faiss
import numpy as np
import threading
import time
from sentence_transformers import SentenceTransformer

class LocalKnowledgeAnchor:
    def __init__(self, model_name='all-MiniLM-L6-v2', dim=384):
        # Initialize sentence transformer running on CPU
        self.encoder = SentenceTransformer(model_name, device='cpu')
        
        # Initialize Faiss index for CPU vector DB
        self.index = faiss.IndexFlatIP(dim) # Inner product for cosine similarity with normalized vectors
        self.chunks = []
        
    def ingest(self, text_chunks):
        if not text_chunks: return
        self.chunks.extend(text_chunks)
        embeddings = self.encoder.encode(text_chunks, normalize_embeddings=True)
        self.index.add(np.array(embeddings, dtype=np.float32))
        
    def retrieve(self, query, k=3):
        if self.index.ntotal == 0:
            return [], []
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(np.array(q_emb, dtype=np.float32), k)
        
        retrieved_chunks = [self.chunks[idx] for idx in indices[0] if idx != -1]
        retrieved_embs = [self.get_chunk_embedding(c) for c in retrieved_chunks]
        return retrieved_chunks, retrieved_embs
        
    def get_chunk_embedding(self, chunk):
        return self.encoder.encode([chunk], normalize_embeddings=True)[0]

class SpeculativeVerifier:
    def __init__(self, anchor: LocalKnowledgeAnchor, threshold=0.85):
        self.anchor = anchor
        self.threshold = threshold
        self.buffer = []
        self.rollback_flag = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self.expected_context_vector = None
        
    def start(self, context_vector):
        self.expected_context_vector = context_vector
        self.buffer = []
        self.rollback_flag = False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._verify_loop)
        self._thread.start()
        
    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            
    def append_to_buffer(self, token_text):
        with self._lock:
            self.buffer.append(token_text)
            
    def check_rollback(self):
        with self._lock:
            if self.rollback_flag:
                self.rollback_flag = False
                return True
            return False

    def _verify_loop(self):
        while not self._stop_event.is_set():
            time.sleep(0.01) # Check buffer periodically
            
            with self._lock:
                current_text = "".join(self.buffer)
                
            if len(current_text.split()) >= 5: # Identify a "Payload Token" block
                # Compute cosine similarity
                current_emb = self.anchor.get_chunk_embedding(current_text)
                sim = np.dot(current_emb, self.expected_context_vector)
                
                if sim < self.threshold:
                    with self._lock:
                        self.rollback_flag = True
                        self.buffer = [] # Reset buffer after trigger
