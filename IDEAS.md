# Research Ideas for Novel LLM Architectures

To outperform Opus locally, we need to push beyond standard "Efficient Transformers." Here are three novel ideas to explore:

### 1. Differential Attention SSMs (Diff-SSM)
Current SSMs (Mamba) struggle with "Associative Recall" (remembering a specific key-value pair from long ago). 
*   **Idea:** Instead of a simple recurrent state, use a "Differential Attention" mechanism where the state update is modulated by the difference between the current input and a compressed "global memory."
*   **Mechanism:** Keep a small, high-precision global state that only updates when the "surprise" (prediction error) is high.

### 2. Recursive Weight Generation (Hyper-BitNet)
Instead of storing all weights for 100B parameters (even at 1.58 bits), use a much smaller "Seed Model" (Hypernetwork) to generate the weights for each layer dynamically.
*   **Idea:** A 1B parameter "Seed" generates the 1.58-bit ternary weights for a virtual 100B model.
*   **Efficiency:** You only store the 1B parameters. Weights are generated on-the-fly or cached in small chunks, drastically reducing RAM usage.

### 3. Logarithmic Quantization MoE
Instead of ternary {-1, 0, 1}, use a logarithmic scale for experts.
*   **Idea:** Experts are quantized to powers of 2 (e.g., {-4, -2, -1, 0, 1, 2, 4}). 
*   **Optimization:** This allows for bit-shift operations instead of additions or multiplications, which are even faster on certain CPU architectures (like ARM).

### 4. Shared-Expert Recurrent MoE
*   **Idea:** Interleave shared experts (active for every token) with specialized experts (routed). The shared experts act as the "working memory" while specialized experts act as "encyclopedic knowledge."
*   **Architecture:** `Token -> SSM -> Shared Expert -> [Routed Experts] -> Output`
