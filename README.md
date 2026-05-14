# Efficient LLM Training & Inference Framework

This project is a research sandbox dedicated to breaking the "Memory Wall" of LLMs. Our goal is to achieve **Claude 3 Opus-level reasoning** on standard local CPUs using a memory footprint of **< 8GB RAM**.

The core philosophy is to move away from dense, high-precision weights toward **Sparse, Ternary, and Generative Flow** architectures.

---

## 📂 Project Structure

```text
├── research/               # Self-contained "Method" directories
│   ├── 01_baseline...      # Baseline BitMamba-MoE
│   ├── 03_hyper_bitnet/    # Memory-winner: Hyper-network based weights
│   └── 10_elf_flow...      # Continuous Flow Matching (Non-autoregressive)
├── src/
│   ├── arch/               # Core model components (Mamba, MoE, Attention)
│   ├── training/           # Standardized evaluation (MMLU, ELF-metrics)
│   └── data/               # Streaming data loaders (OpenWebText, Alpaca)
└── scripts/                # Benchmark and utility scripts
```

---

## 🧠 Methodologies & Inspirations

### 1. ELF: Embedded Language Flows (`research/10`)
**Main Inspiration:** *ELF: Embedded Language Flows (arXiv:2605.10938)*

Unlike standard Autoregressive (AR) models that predict tokens one-by-one, ELF treats text as a continuous trajectory in embedding space.
*   **Optimal Transport (Efficiency Add-on):** We implement **Path-Energy Regularization**, which penalizes high-velocity movements in the latent space. This forces the model to find the "Least Action" (straightest) path between noise and text, improving convergence speed.
*   **Self-Conditioning:** The model uses its previous clean-text predictions to stabilize the current flow, preventing "drift" during generation.
*   **SDE Sampling:** We use a Stochastic Differential Equation (SDE) solver for inference, injecting decaying noise to explore the probability manifold more effectively than simple ODE solvers.

### 2. Hyper-BitNet (`research/03`)
**Main Inspiration:** *BitNet b1.58 + HyperNetworks*

This is currently our **benchmark winner** for memory efficiency.
*   **Dynamic Weight Generation:** Instead of storing billions of parameters, a small "Seed" model (HyperNetwork) generates the 1.58-bit (ternary) weights on the fly.
*   **Ternary Arithmetic:** All linear operations are performed using `{-1, 0, 1}` values, allowing for addition-only matrix multiplications that are highly optimized for CPU SIMD instructions.

### 3. BitMamba-MoE (The Backbone)
**Inspiration:** *Mamba (SSM) + Mixture of Experts (MoE)*

The project uses a hybrid backbone that combines:
*   **SSM (State Space Models):** Provides $O(N)$ context scaling with a constant-size KV cache, replacing the memory-heavy $O(N^2)$ Self-Attention.
*   **Shared-Expert MoE:** Uses routed experts to provide high model capacity while only activating a small fraction of the total parameters for each token.

---

## 🚀 How to Run

Each method is self-contained in the `research/` folder and follows a standardized logging and evaluation flow.

### Running an Experiment (e.g., ELF)
```bash
# Set PYTHONPATH to the root
export PYTHONPATH=.

# Run the production training (10k steps)
./.venv/bin/python3 research/10_elf_flow_matching/experiment.py
```

### Monitoring & Results
*   **Real-time Logs:** Each method outputs to its own `training.log` (e.g., `research/10_elf_flow_matching/training.log`).
*   **Metrics:** Final scores for Generative Perplexity (Gen. PPL) and Unigram Entropy are saved to `results.json` in the respective method directory.

---

## 📊 Results Summary
| Method | Key Advantage | Final Loss (300 steps) |
| :--- | :--- | :--- |
| **Efficient ELF** | Direct Path / Stable Decoding | 1.63 (Total) |
| **Hyper-BitNet** | 100x Memory Compression | 4.29 (Flow) |
| **Baseline** | Stable Convergence | 7.80 (Flow) |

Full benchmarking details are available in [RESULTS.md](./RESULTS.md).
