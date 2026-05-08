# Balanced Bilateral Routing (BBR)

A novel router for Mixture-of-Experts (MoE) Transformers that balances expert utilization via prototype‑based token assignment with Sinkhorn balancing.

## Overview

This project compares two routing strategies for MoE layers:

- **Standard Router** – learned linear gate with softmax + argmax top‑1 routing, plus an auxiliary load‑balancing loss (Z‑loss).
- **BBR Router** – prototype‑based routing where tokens are projected into a low‑dimensional embedding space and matched to expert prototypes via Sinkhorn‑balanced assignment. Features dynamic capacity allocation based on per‑expert confidence EMA and a collapse loss to keep prototypes diverse.

## Architecture

- 2‑layer Transformer with causal masking (4 heads, hidden size 256)
- 8 FFN experts (inner dimension 512)
- Trained on WikiText‑103 (tokenized with GPT‑2 tokenizer, sequence length 128)
- 10k training steps, cosine LR schedule, AdamW

## Running the Experiment

Open `balanced_bileteral_routing.ipynb` in Google Colab (GPU runtime required). The notebook:

1. Installs dependencies (`torch`, `transformers`, `datasets`)
2. Downloads WikiText‑103
3. Trains both routers sequentially (~20 min each)
4. Logs perplexity and expert utilization entropy every 500 steps
5. Saves results to `bbr_results.json`