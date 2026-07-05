# Gemma3-500M

A Gemma 3-inspired 500M parameter decoder-only language model built entirely from scratch in PyTorch.

This project started as an experiment after implementing a small 14M parameter transformer trained on Tiny Shakespeare. Once that was working, the goal shifted towards reproducing many of the architectural ideas behind modern open-weight LLMs while keeping the implementation understandable and relatively lightweight.

Every major component from the transformer architecture to the distributed training pipeline was written from scratch.

The project is still a work in progress. Training currently becomes numerically unstable after roughly 3–4K optimization steps, so the model is **not** fully trained yet. Even so, the repository contains the complete architecture, training pipeline, checkpointing system, and infrastructure required to continue experimentation.

**Checkpoints**: https://huggingface.co/notninja/gemma-500m-smollm

---

## Model

| Property            |                             Value |
| ------------------- | --------------------------------: |
| Parameters          |                             ~500M |
| Architecture        |          Decoder-only Transformer |
| Layers              |                                28 |
| Hidden Size         |                              1024 |
| Attention Heads     |                                16 |
| KV Heads            |       4 (Grouped Query Attention) |
| Context Length      |                              2048 |
| Activation          |                 Approximate GeGLU |
| Normalization       |                           RMSNorm |
| Positional Encoding | Rotary Position Embeddings (RoPE) |
| Attention           | Sliding Window + Global Attention |
| Precision           |                          bfloat16 |
| Training            |   Distributed Data Parallel (DDP) |

---

## Features

* Gemma 3-inspired architecture
* RoPE positional embeddings
* Grouped Query Attention (GQA)
* Sliding-window attention with periodic global attention layers
* Pre-Norm transformer blocks using RMSNorm
* Approximate GeGLU feed-forward network
* FlashAttention-2 integration
* Weight tying between token embeddings and LM head
* Gradient checkpointing
* Cosine learning rate schedule with warmup
* Gradient clipping
* Mixed precision (bfloat16)
* Distributed training across 4× NVIDIA A10G GPUs
* Streaming dataset pipeline
* Automatic checkpoint uploads to Hugging Face
* Weights & Biases logging

---

## Training

The model is trained on the **HuggingFaceTB/SmolLM Cosmopedia** dataset using a streaming data pipeline to avoid storing the full dataset locally.

The training stack includes:

* PyTorch 2.x
* FlashAttention-2
* Hugging Face Datasets
* Distributed Data Parallel
* Gradient accumulation
* Cosine LR decay
* Automatic checkpoint recovery
* Background checkpoint uploads

Training was performed on **4× NVIDIA A10G GPUs** using Modal.

---

## Repository Structure

```
.
├── model.py              # Transformer architecture
├── train.py              # Distributed training pipeline
└── README.md
```

---

## Current Status

The implementation is complete, but training is currently unstable.

The loss decreases normally for the first few thousand optimization steps before gradients suddenly explode. The exact cause is still under investigation and could be related to numerical instability, distributed synchronization, the streaming/tokenization pipeline, or another implementation detail.

Because of limited compute credits, I wasn't able to continue debugging further.

Contributions, suggestions, or investigations into the instability are always welcome.

---

## Future Work

* Stabilize long-running pretraining
* Add inference KV cache
* Support longer context lengths
* Improve checkpoint recovery
* Add evaluation benchmarks
* Add instruction tuning pipeline
* Experiment with larger model sizes

---

## Running Training

```bash
modal run train.py
```

The training script automatically:

* launches Distributed Data Parallel workers
* streams the dataset
* logs metrics to Weights & Biases
* periodically saves checkpoints
* uploads checkpoints to Hugging Face

---

## Acknowledgements

This project was heavily inspired by modern open language models including Gemma, Llama, Mistral, and SmolLM, along with educational resources from Andrej Karpathy and the open-source AI community.

The implementation itself, however, was written from scratch as a learning project to better understand how modern LLMs are built and trained.
