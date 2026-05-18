# NVIDIA PyTorch 25.04: PyTorch 2.8, CUDA 12.8, Python 3.12
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-04.html
FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV MAX_JOBS=32
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""

RUN apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-confdef" systemd tini && \
    apt-get clean || { echo "Installation failed"; exit 1; }

RUN apt-get install -y tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir \
    "transformers>=4.54.1" accelerate datasets peft hf-transfer diffusers \
    tensordict torchdata \
    codetiming hydra-core pandas "pyarrow>=15.0.0" pylatexenc qwen-vl-utils wandb ninja "liger-kernel==0.5.8" \
    "flash-linear-attention>=0.2.0" \
    pytest yapf py-spy pre-commit ruff packaging

RUN pip install --no-cache-dir --no-build-isolation flash-attn
