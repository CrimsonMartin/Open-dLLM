# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open-dLLM is a diffusion-based large language model framework for training, evaluation, and inference. It converts autoregressive LMs (Qwen, LLaMA, DeepSeek) into discrete diffusion LMs, offering ~4x speedup. The core package is `veomni/`.

## Commands

```bash
# Setup
pip install -e ".[dev]"

# Linting & formatting (ruff, line-length=119, target py38)
make style          # auto-fix
make quality        # check only

# Tests
make test           # pytest tests/

# Pre-commit
make commit         # pre-commit run --all-files
```

No single-file test runner is configured; use `pytest tests/path/test_file.py` directly.

## Architecture

### Training Pipeline (`tasks/train_torch.py`)
Single entry point for all training. Configs are YAML files consumed by a Hydra-style argument parser (`veomni/utils/arguments.py`). Three dataclass groups: `ModelArguments`, `DataArguments`, `TrainingArguments`.

- **Pretraining configs**: `configs/pretrain/` (plaintext datasets, FSDP1 or DDP)
- **SFT configs**: `configs/sft/` (conversation data, DeepSeek MoE support)
- **Multimodal configs**: `configs/multimodal/` (vision-language, omni-modal, representation alignment)

### Model Implementations (`veomni/models/transformers/`)
Each model family is a subpackage with its own `modeling_*.py` and optional `generation_utils.py`:
- **qwen2** — base autoregressive (Qwen2-0.5B/7B/32B/72B)
- **qwen2_vl** / **qwen2_5vl** — vision-language variants
- **qwen3** — Qwen3 (newer generation)
- **qwen3_5** — Qwen3.5/3.6 architecture with hybrid linear/full attention (Gated DeltaNet)
- **qwen3_5_moe** — Qwen3.5/3.6 MoE variant (256 experts, shared expert, expert parallelism)
- **llama** — LLaMA3-8B/72B
- **deepseek_v3** — MoE models with routed experts

New models are registered in `veomni/models/transformers/__init__.py`. Architecture JSON configs live in `configs/model_configs/{family}/`.

### Seed Omni (`veomni/models/seed_omni/`)
Multi-modal foundation model combining encoders (e.g., Qwen2-VL vision) with decoders (e.g., MOVQGAN). Built via `build_omni_model()`.

### Distributed Training (`veomni/distributed/`)
- **FSDP1**: full-shard data parallel via PyTorch FSDP
- **DDP**: standard distributed data parallel
- **Sequence parallel (Ulysses)**: `veomni/distributed/sequence_parallel/` — splits long sequences across GPUs
- **MoE**: `veomni/distributed/moe/` — expert parallelism, fused MoE kernels
- **Parallel plan**: `parallel_plan.py` / `vescale_plan.py` define sharding strategies

### Data (`veomni/data/`)
Supports both plaintext and conversation formats. Key: `build_mapping_dataset()` (map-style), `build_iterative_dataset()` (iterable/streaming). Dynamic batching via `dynamic_batching.py`.

### Loss Functions (`veomni/ops/loss.py`)
Implements cross-entropy losses with fused kernel support: `seed_kernels` > `liger-kernel` > vanilla fallback.

### Checkpointing (`veomni/checkpoint/`)
Primary manager is `bytecheckpoint` with DCP (Distributed Checkpoint) format. `mereg_dcp_to_hf.py` script converts to HF format.

## Evaluation

- **Code completion**: `eval/eval_completion/` — uses lm-evaluation-harness (HumanEval, MBPP)
- **Code infilling**: `eval/eval_infill/` — uses torchrun with DDP
- Both use `accelerate launch` or `torchrun` with custom diffusion generation

## Key Patterns

- Models are loaded via `veomni/models/auto.py`: `build_foundation_model(config_path, weights_path, ...)` which dispatches to per-family loaders in `veomni/models/loader.py`
- Diffusion generation uses `model.diffusion_generate()` with `MDMGenerationConfig` (mask tokens, steps, algorithm selection like `p2`)
- All model classes use `trust_remote_code=True`
- Config files reference HDFS paths for ByteDance internal clusters; local development uses HF model paths
- Representation alignment (repr_align) trains a diffusion LM by aligning to autoregressive teacher representations — configured via `repr_align_wt` in training YAMLs
