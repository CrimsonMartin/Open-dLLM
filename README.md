
# Open-dLLM: Open Diffusion Large Language Models

> **Fork focus**: LDLM (Latent Diffusion Language Model) training on consumer GPUs, with speculative decoding support. Based on [Open-dLLM](https://github.com/pengzhangzhi/Open-dLLM) and the [LDLM paper](https://arxiv.org/abs/2605.07933).

---

## What This Fork Adds

This fork extends Open-dLLM with a complete **LDLM training and speculative decoding pipeline** that runs on a single consumer GPU (RTX 3090, 24GB).

**Key contributions:**

- **LDLM training script** (`tasks/train_ldlm.py`) — full implementation of the LDLM paper's training recipe: Perceiver autoencoder, DiT diffusion head, adaptive timestep sampling, sigmoid warmup, tangent noise schedule, self-conditioning, and EMA-based latent normalization.
- **Batch unpacking fix** — the upstream data pipeline packs all sequences into `[1, N]` (batch dim always 1). We reshape packed inputs to `[B, seq_len]` inside the forward pass, giving real batch gradients. This is critical — B=1 causes mode collapse; B=64+ trains stably.
- **Cross-attention prefix conditioning** — the diffusion head cross-attends to frozen encoder hidden states from a prefix, so it learns to generate *continuations* rather than unconditional samples. Each training sample is split into prefix (context) and target (what gets denoised).
- **Speculative decoding pipeline** (`tasks/spec_decode_ldlm.py`) — draft K tokens in parallel via latent diffusion, verify against the target AR model in a single forward pass. Supports cross-attention conditioning on target model hidden states.
- **Consumer GPU configs** — training configs for Qwen2.5-0.5B on a single RTX 3090 with the frozen encoder on GPU.

### Architecture

```
Frozen Encoder (Qwen2.5-0.5B)
    |
    v
Hidden states h  -->  [prefix h]  cross-attention context
    |                      |
    v                      v
Perceiver Encoder    DiffusionHead (DiT + cross-attn)
    |                      |
    v                      v
  z0 (latents)       denoised z0_hat
    |
    v
Perceiver Decoder --> LM Head --> token predictions
```

During **training**, the frozen encoder produces hidden states for the full input. The prefix half provides cross-attention context; the target half is compressed by the Perceiver into latents, which the diffusion head learns to denoise. Reconstruction losses (MSE on hidden states + CE on tokens) train the autoencoder; diffusion MSE trains the denoising head.

During **speculative decoding**, the diffusion head generates K draft tokens in parallel (conditioned on prefix hidden states from the target model), and the AR target model verifies them in one forward pass.

---

## Quickstart: LDLM Training

### 1. Setup

```bash
pip install -e ".[dev]"
```

### 2. Prepare Data

Tokenize plaintext data into parquet, then the training script auto-converts to memory-mapped format:

```bash
python tasks/build_parquet.py \
  --input data/fineweb_5m.jsonl \
  --tokenizer Qwen/Qwen2.5-0.5B \
  --max_seq_len 16 \
  --output data/fineweb_5m.tokenized.parquet
```

### 3. Train

```bash
# Single GPU (RTX 3090 / 4090 / A6000)
torchrun --nproc_per_node=1 tasks/train_ldlm.py \
  configs/pretrain/qwen2_5_05b_ldlm_v12.yaml
```

The v12 config trains with:
- `seq_len=8` (8 latent tokens per sample)
- `max_seq_len=16` (8 prefix + 8 target, enabling cross-attention)
- `micro_batch_size=256` (B=256 real samples after unpacking)
- Frozen Qwen2.5-0.5B encoder on GPU
- ~15GB VRAM estimated

Generation eval runs every 1000 steps and prints samples to the log.

### 4. Speculative Decoding

```bash
python tasks/spec_decode_ldlm.py \
  --checkpoint_path path/to/ldlm_checkpoint \
  --target_model Qwen/Qwen2.5-0.5B \
  --prompt "def quicksort(arr):" \
  --draft_k 8 \
  --draft_steps 10 \
  --cross_attn \
  --baseline  # also runs vanilla AR for speed comparison
```

---

## Training Configs

| Config | seq_len | max_seq_len | Batch (effective) | Cross-attn | Notes |
|--------|---------|-------------|-------------------|------------|-------|
| `qwen2_5_05b_ldlm_v11.yaml` | 8 | 8 | B=64 | Off (no prefix room) | Minimal config, fast iteration |
| `qwen2_5_05b_ldlm_v12.yaml` | 8 | 16 | B=256 | On (8 prefix + 8 target) | Full pipeline, recommended |

Both use constant LR (1e-4), AdamW, bf16 mixed precision, and DDP.

---

## Key Implementation Details

### Batch Unpacking

The upstream dataloader packs sequences via `rmpad_with_pos_ids`, producing `[1, total_tokens]` tensors. Our `ldlm_forward()` reshapes this to `[B, sample_len]` before the forward pass:

```python
# [1, 4096] -> [256, 16] for B=256 with sample_len=16
if B_in == 1 and T_in > seq_len:
    n_samples = T_in // sample_len
    input_ids = input_ids[0, :n_samples * sample_len].reshape(n_samples, sample_len)
```

Without this, the diffusion head trains on a single sample per step, leading to mode collapse (generates "the the the the...").

### Staged Autoencoder Encoding

The autoencoder exposes `encode_hidden()` (frozen encoder + h normalization) and `encode_latent()` (Perceiver + z0 normalization) separately. This lets us run the frozen encoder on the full input (prefix + target), then split, and run the Perceiver on target tokens only.

### Cross-Attention in DiffusionHead

Each DiT block has optional cross-attention with a zero-initialized gate:

```python
if self.has_cross_attn and context is not None:
    x = x + self.cross_gate * self.cross_attn(self.norm_cross(x), context=context)
```

The gate starts at zero (no disruption to pretrained weights) and learns to open during training.

---

## Original Open-dLLM

This fork is based on [Open-dLLM](https://github.com/pengzhangzhi/Open-dLLM), the most open release of a diffusion-based large language model — including pretraining, evaluation, inference, and checkpoints.

Open-dLLM also supports:
- **Masked Diffusion Models** (MDM) for code generation (Open-dCoder 0.5B)
- **Representation alignment** for adapting AR models to diffusion models
- **Full evaluation suite** (HumanEval, MBPP, code infilling)

See the [upstream README](https://github.com/pengzhangzhi/Open-dLLM) for the complete documentation on these features.

---

## Citation

```bibtex
@misc{opendllm2025,
  title        = {Open-dLLM: Open Diffusion Large Language Models},
  author       = {Fred Zhangzhi Peng, Shuibai Zhang, Alex Tong, and contributors},
  year         = {2025},
  howpublished = {\url{https://github.com/pengzhangzhi/Open-dLLM}},
}

@article{ldlm2025,
  title   = {Latent Diffusion Language Models},
  author  = {Guangyi Liu and others},
  year    = {2025},
  journal = {arXiv preprint arXiv:2605.07933},
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
