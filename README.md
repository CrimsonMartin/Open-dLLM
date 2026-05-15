
# 🔥 Open-dLLM: Open Diffusion Large Language Models

🌍 Languages: [English](README.md) | [中文](README_cn.md) | [日本語](README_ja.md)

👉 TL;DR: **Open-dLLM** is the most open release of a diffusion-based large language model to date —  
including **pretraining, evaluation, inference, and checkpoints**.  


#### Representation Alignment

Open-dLLM supports **representation alignment** for adapting autoregressive LMs into diffusion LMs with 4x speedup. This feature is based on our recent paper, [**Don’t Retrain—Align: Adapting Autoregressive LMs to Diffusion LMs via Representation Alignment**](https://arxiv.org/pdf/2605.06885). Check out [Representation Alignment Tutorial](docs/representation_alignment.md).


<p align="center">
  <a href="https://github.com/pengzhangzhi/Open-dLLM">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="40" alt="GitHub"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://oval-shell-31c.notion.site/Open-Diffusion-Large-Language-Model-25e03bf6136480b7a4ebe3d53be9f68a?pvs=74">
    <img src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Notion-logo.svg" width="40" alt="Notion"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/fredzzp/open-dcoder-0.5B">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="40" alt="Hugging Face"/>
  </a>
</p>

<p align="center">
  <b>💻 Code</b> &nbsp; | &nbsp; <b>📖 Blog</b> &nbsp; | &nbsp; <b>🤗 Model</b>
</p>


## 🎥 Demo

<p align="center">
  <img src="https://github.com/pengzhangzhi/dLLM-training/blob/main/assets/quick-sort-demo.gif" 
       alt="Quick Sort Demo" width="600"/>
</p>

<p align="center"><i>QuickSort generation using Open-dCoder (0.5B)</i></p>

<p align="center">
  <a href="https://youtu.be/d8WrmvUhO9g">
    <img src="https://img.shields.io/badge/YouTube-Video-red?logo=youtube" alt="YouTube link"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.bilibili.com/video/BV1ZveSz3E1J/">
    <img src="https://img.shields.io/badge/Bilibili-视频-blue?logo=bilibili" alt="Bilibili link"/>
  </a>
</p>

---

## ✨ Highlights

- 🏋️ **Pretraining pipeline + open datasets**  
- ⚡ **Inference scripts** — easy sampling & generation  
- 📊 **Evaluation suite** — HumanEval, MBPP, Infilling (lm-eval-harness + custom metrics)  
- 📦 **Weights + checkpoints** on Hugging Face  
- 🤝 **Transparent configs** for full reproducibility  

---

## Why Open-dLLM?

Most diffusion LLM repos (e.g., LLaDA, Dream) only release **inference scripts + weights**, which limits reproducibility.  
**Open-dLLM** is the first to open-source the **entire stack** for diffusion LLMs.

👉 With Open-dLLM, you can go from **raw data → training → checkpoints → evaluation → inference**, all in one repo.

---

## 🔎 Transparency Comparison of Diffusion LLM Releases

| Project                                                                 | Data | Training Code | Inference | Evaluation | Weights |
|-------------------------------------------------------------------------|:---:|:-------------:|:---------:|:----------:|:-------:|
| **Open-dLLM / Open-dCoder (ours)**                                      | ✅  | ✅            | ✅        | ✅         | ✅      |
| [LLaDA](https://github.com/ML-GSAI/LLaDA)                               | ❌  | ❌            | ✅        | ⚠️ Limited | ✅      |
| [Dream](https://github.com/HKUNLP/Dream)                                | ❌  | ❌            | ✅        | ⚠️ Limited | ✅      |
| [Gemini-Diffusion](https://deepmind.google/models/gemini-diffusion/)    | ❌  | ❌            | ❌        | ❌         | ❌ (API only) |
| [Seed Diffusion](https://seed.bytedance.com/seed_diffusion)             | ❌  | ❌            | ❌        | ❌         | ❌ (API only) |
| [Mercury](https://www.inceptionlabs.ai/introducing-mercury-our-general-chat-model) | ❌  | ❌            | ❌        | ❌         | ❌ (API only) |

✅ = fully available · ❌ = not provided · ⚠️ = partial/limited

---

## ⚙️ Install

We use `micromamba` for environment management (feel free to adapt to `conda`):

```bash
micromamba install -c nvidia/label/cuda-12.3.0 cuda-toolkit -y
pip install ninja

# install the newest torch with cu121
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121

pip install "flash-attn==2.7.4.post1" \
  --extra-index-url https://github.com/Dao-AILab/flash-attention/releases/download

pip install --upgrade --no-cache-dir \
  tensordict torchdata triton>=3.1.0 \
  transformers==4.54.1 accelerate datasets peft hf-transfer \
  codetiming hydra-core pandas pyarrow>=15.0.0 pylatexenc \
  wandb ninja liger-kernel==0.5.8
# optional
pip install pytest yapf py-spy pyext pre-commit ruff packaging

pip install -e .
pip install lm-evaluation-harness/ human-eval-infilling/
````

---

## 🚀 Quickstart: Sampling

```python
from transformers import AutoTokenizer
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig
import torch

model_id = "fredzzp/open-dcoder-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device).eval()

# Prompt
prompt = "Write a quick sort algorithm in python."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generation config
gen_cfg = MDMGenerationConfig(max_new_tokens=128, steps=200, temperature=0.7)

with torch.no_grad():
    outputs = model.diffusion_generate(inputs=input_ids, generation_config=gen_cfg)

print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
```

👉 For full logging, history tracking, and file output:

```bash
python sample.py
```

---

## 📊 Benchmarking

We release a fully open-source **evaluation suite** for diffusion-based LLMs (dLLMs), covering both **standard code generation tasks** and **code infilling tasks**.

Benchmarks include: **HumanEval / HumanEval+**, **MBPP / MBPP+**, **HumanEval-Infill**, **SantaCoder-FIM**.

---

#### Standard Code Generation

| Method                       | HumanEval |          | HumanEval+ |          | MBPP     |          | MBPP+    |          |
| ---------------------------- | --------- | -------- | ---------- | -------- | -------- | -------- | -------- | -------- |
|                              | Pass\@1   | Pass\@10 | Pass\@1    | Pass\@10 | Pass\@1  | Pass\@10 | Pass\@1  | Pass\@10 |
| LLaDA (8B)                   | 35.4      | 50.0     | 30.5       | 43.3     | 38.8     | 53.4        | 52.6     | 69.1        |
| Dream (7B)                   | 56.7      | 59.2     | 50.0       | 53.7     | 55.4     | 56.2        | 71.5     | 72.5        |
| Mask DFM (1.3B)              | 9.1       | 17.6     | 7.9        | 13.4     | 6.2      | 25.0     | –        | –        |
| Edit Flow (1.3B)             | 12.8      | 24.3     | 10.4       | 20.7     | 10.0     | 36.4     | –        | –        |
| **Open-dCoder (0.5B, Ours)** | **20.8**  | **38.4** | **17.6**   | **35.2** | **16.7** | **38.4** | **23.9** | **53.6** |

> *Despite being only 0.5B parameters, Open-dCoder competes with much larger dLLMs in code completion tasks.*

---

#### Code Infilling

| Method                                | HumanEval Infill Pass@1 | SantaCoder Exact Match |
| ------------------------------------- | ----------------------: | ---------------------: |
| LLaDA-8B                              |                    48.3 |                  35.1  |
| Dream-7B                              |                    39.4 |                  40.7  |
| DiffuCoder-7B                         |                    54.8 |                  38.8  |
| Dream-Coder-7B                        |                    55.3 |                  40.0  |
| **Open-dCoder (0.5B, Ours)**          |                    32.5 |                  29.6  |
| **Open-dCoder (0.5B, Ours)** Oracle Length |               77.4 |                  56.4  |

> *We followed the average fixed length evaluation setting in [DreamOn](https://hkunlp.github.io/blog/2025/dreamon/) to get the results.*

---

## 🧪 Evaluation

Install evaluation packages:

```bash
pip install -e lm-evaluation-harness human-eval-infilling
```

#### Code Completion (HumanEval, MBPP)

```bash
cd eval/eval_completion
bash run_eval.sh
```

#### Code Infilling

```bash
cd eval/eval_infill
bash run_eval.sh
```

---

## 🏋️ Pretraining

* **Data**: Concise, high-quality code corpus [**FineCode**](https://huggingface.co/datasets/fredzzp/fine_code), hosted on Hugging Face.
* **Initialization**: Following *Dream*, continued pretraining from **Qwen2.5-Coder**, adapting it into the diffusion framework.
* **Loss**: Masked Diffusion Model (MDM) objective — masking ratios uniformly sampled from `[0,1]`, reconstructed with cross-entropy loss.

### Download Data

```bash
python3 scripts/download_hf_data.py --repo_id fredzzp/fine_code --local_dir ./data
```

### Training

```bash
export TOKENIZERS_PARALLELISM=false
NNODES=1
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}



torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT tasks/train_torch.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --data.train_path=data/data \
  --train.ckpt_manager=dcp \
  --train.micro_batch_size=16 \
  --train.global_batch_size=512 \
  --train.output_dir=logs/Qwen2.5-Coder-0.5B_mdm \
  --train.save_steps=10000
```
example of multi-node training with repr alignment loss:
```bash

export TOKENIZERS_PARALLELISM=false

NNODES=${NNODES:=1}
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}
torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK   --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT  tasks/train_torch.py \
configs/pretrain/qwen2_5_coder_500M.yaml --data.train_path=data/data \
--data.num_workers=0 \
--data.prefetch_factor=1 \
--train.ckpt_manager=dcp \
--train.micro_batch_size=3 \
--train.global_batch_size=240 \
--train.repr_align_wt=10.0 \
--model.model_path=Qwen/Qwen2.5-Coder-3B-Instruct \
--train.save_steps=10000 \
--train.output_dir=logs/Qwen2.5-Coder-3B-Instruct_mdm_repr_align-10
```

### Uploading Checkpoints to Hugging Face

```python
from huggingface_hub import HfApi

REPO_ID = "fredzzp/open-dcoder-0.5B"
LOCAL_DIR = "logs/Qwen2.5-Coder-0.5B_mdm/checkpoints/global_step_370000/hf_ckpt"

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=REPO_ID, repo_type="model", folder_path=LOCAL_DIR)
```

---

## 🧬 LDLM: Latent Diffusion Language Model

Open-dLLM supports **LDLM** (Latent Diffusion Language Model) — a Perceiver-based latent diffusion approach that compresses large autoregressive model representations into a compact latent space for diffusion-based generation. This enables training diffusion LMs on top of frozen large models (e.g., Qwen3.6-27B) with minimal trainable parameters.

#### Recreating the Benchmarks

```bash
# 1. Install dependencies (see Install section above)
pip install -e .

# 2. Download the encoder model (only needed for training; benchmark downloads automatically)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.6-35B-A3B')   # ~22GB download
# snapshot_download('Qwen/Qwen3.6-27B')     # ~54GB download
"

# 3. Run inference benchmark on a single GPU
CUDA_VISIBLE_DEVICES=0 python tasks/benchmark_ldlm_35b.py    # Qwen3.6-35B-A3B
CUDA_VISIBLE_DEVICES=0 python tasks/benchmark_ldlm.py        # Qwen3.6-27B
```

> **Hardware used**: NVIDIA RTX 5090 (32GB VRAM), 91GB system RAM, Python 3.13, PyTorch 2.7+, CUDA 12.x.

#### Inference Throughput (Qwen3.6 LDLM, untrained, RTX 5090 32GB)

| Model | Dim | Trainable Params | Diffusion Steps | Throughput |
|-------|-----|-------------------|-----------------|------------|
| Qwen3.6-35B-A3B | 2048 | 1.39B | 10 | **3,238 tok/s** |
| Qwen3.6-35B-A3B | 2048 | 1.39B | 4 | **~6,500 tok/s** |
| Qwen3.6-27B | 5120 | 6.75B | 10 | **745 tok/s** |
| Qwen3.6-27B | 5120 | 6.75B | 4 | **~1,500 tok/s** |

> For comparison, autoregressive generation on the same hardware achieves ~30-50 tok/s for a 27B model.

#### Assumptions & Caveats

- **Untrained weights**: These benchmarks use randomly initialized Perceiver/decoder/diffusion-head weights. A trained model will have identical throughput but produce coherent output. Quality benchmarks (perplexity, HumanEval) will be published after training completes.
- **No encoder in the loop**: The frozen Qwen3.6 encoder is **not used during generation** — it's only needed for training (to produce latent targets). At inference, the diffusion head denoises random noise, then the Perceiver decoder maps latents to tokens. The encoder is deleted before benchmarking (`del autoencoder.token_encoder`).
- **Seq len = 64**: The benchmark uses a short sequence length (64 tokens). Longer sequences will reduce throughput proportionally. The 4-step throughput numbers are linear extrapolations from the 10-step measurements.
- **Batch size = 1**: Single-sequence generation only. Throughput scales near-linearly with batch size for the 35B-A3B (dim=2048 fits easily in VRAM), less so for the 27B (dim=5120).
- **CPU RAM requirement**: While the encoder is not used at inference, it **must** fit in system RAM during training (~54GB for 27B, ~22GB for 35B-A3B in bf16). The Qwen3.6 architecture uses Triton kernels (flash-linear-attention) that cannot run on CPU, so the encoder forward pass during training requires GPU offloading — a multi-GPU setup is recommended for training.
- **Qwen3.6 requires `trust_remote_code=True`**: The model uses custom architecture code (`Qwen3_5ForConditionalGeneration`) that is not in standard transformers releases. Ensure your `transformers` version supports it (>=4.54).
- **35B-A3B is MoE**: Only 3B of its 35B parameters are active per token, giving it a much smaller hidden dim (2048) than the 27B dense model (5120). This is why the LDLM trainable components are 5x smaller and 4x faster.
- **Not an apples-to-apples comparison with AR models**: The diffusion model generates all tokens in parallel across N diffusion steps, while AR generates one token at a time. The "tok/s" metric favors diffusion for short sequences but does not reflect output quality, which depends on training convergence.

#### How to Train a Qwen3.6 LDLM

1. **Download the base model** (27B dense or 35B-A3B MoE):
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.6-27B', local_dir='./qwen36_27b_local')
# or for MoE:
# snapshot_download('Qwen/Qwen3.6-35B-A3B', local_dir='./qwen36_35b_a3b_local')
"
```

2. **Prepare training data** (e.g., FineWeb):
```bash
python -c "
from datasets import load_dataset
import json
ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=True)
with open('data.jsonl', 'w') as f:
    for i, ex in enumerate(ds):
        if i >= 100000: break
        f.write(json.dumps({'text': ex['text']}) + '\n')
"
```

3. **Run the benchmark** (verify setup before training):
```bash
# 27B
CUDA_VISIBLE_DEVICES=0 python tasks/benchmark_ldlm.py
# 35B-A3B MoE
CUDA_VISIBLE_DEVICES=0 python tasks/benchmark_ldlm_35b.py
```

4. **Start training**:
```bash
# 27B
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tasks/train_ldlm.py \
  configs/pretrain/qwen3_6_27b_ldlm.yaml
# 35B-A3B MoE
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tasks/train_ldlm.py \
  configs/pretrain/qwen3_6_35b_a3b_ldlm.yaml
```

> **GPU Memory**: The frozen encoder runs on CPU (~54GB RAM for 27B, ~22GB for 35B-A3B). Only trainable components run on GPU. The 35B-A3B MoE variant has a smaller hidden dim (2048 vs 5120), making it significantly faster and more memory-efficient — ideal for consumer GPUs.

---

## 🙏 Appreciation

This project builds on incredible prior work:

* **Frameworks & Tooling**: [VeOmni](https://github.com/ByteDance-Seed/VeOmni), [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
* **Open-source dLLMs**: [LLaDA](https://github.com/ML-GSAI/LLaDA), [Dream](https://github.com/HKUNLP/Dream)
* **Pioneering dLLMs**: [Gemini-Diffusion](https://deepmind.google/models/gemini-diffusion/), [Seed Diffusion](https://seed.bytedance.com/seed_diffusion), [Mercury](https://www.inceptionlabs.ai/introducing-mercury-our-general-chat-model)
* **Foundational research**: [MD4](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html), [MDLM](https://arxiv.org/abs/2406.07524), [DPLM](https://github.com/bytedance/dplm)

We stand on the shoulders of these projects, and hope Open-dLLM contributes back to the diffusion LLM community.




## 📚 Citation

If you use **Open-dLLM** or **Open-dCoder** in your research, please cite us:

```bibtex
@misc{opendllm2025,
  title        = {Open-dLLM: Open Diffusion Large Language Models},
  author       = {Fred Zhangzhi Peng, Shuibai Zhang, Alex Tong, and contributors},
  year         = {2025},
  howpublished = {\url{https://github.com/pengzhangzhi/Open-dLLM}},
  note         = {Blog: \url{https://oval-shell-31c.notion.site/Open-Diffusion-Large-Language-Model-25e03bf6136480b7a4ebe3d53be9f68a?pvs=74}, 
                  Model: \url{https://huggingface.co/fredzzp/open-dcoder-0.5B}}
}
