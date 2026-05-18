# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LDLM (Latent Diffusion Language Model) training script.

Reuses Open-dLLM infrastructure (dataloader, optimizer, FSDP, checkpointing)
while implementing the LDLM-specific forward pass with adaptive timestep
sampling, latent diffusion, and reconstruction losses.

Usage:
    torchrun --nproc_per_node=8 tasks/train_ldlm.py --config configs/pretrain/qwen3_6_27b_ldlm.yaml
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.data.data_transform import process_pretrain_example, process_sft_example
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.models.ldlm import (
    LDLMAutoencoder,
    DiffusionHead,
    AdaptiveTimestepSampler,
    LDLMTrainer as LDLMTrainerWrapper,
)
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.dist_utils import all_reduce


logger = helper.create_logger(__name__)


KEEP_LAST_CHECKPOINTS = 2
"""Number of latest checkpoints to keep. Older ones are pruned to save disk."""


def prune_checkpoints(save_dir: str, keep: int = KEEP_LAST_CHECKPOINTS):
    """
    Remove all but the latest `keep` checkpoint directories in save_dir.
    Only prunes on rank 0 to avoid races.
    """
    if dist.get_rank() != 0:
        return
    if not os.path.isdir(save_dir):
        return
    ckpt_dirs = [
        d for d in os.listdir(save_dir)
        if d.startswith("global_step_") and os.path.isdir(os.path.join(save_dir, d))
    ]
    ckpt_dirs.sort(key=lambda d: int(d.replace("global_step_", "")))
    for old in ckpt_dirs[:-keep]:
        path = os.path.join(save_dir, old)
        try:
            import shutil
            shutil.rmtree(path)
            logger.info(f"Pruned old checkpoint: {path}")
        except Exception as e:
            logger.warning(f"Failed to prune {path}: {e}")
    # Also prune eval checkpoints (keep last 1)
    eval_dirs = [
        d for d in os.listdir(save_dir)
        if d == "eval" and os.path.isdir(os.path.join(save_dir, d))
    ]
    if len(eval_dirs) > 1:
        for old in eval_dirs[:-1]:
            path = os.path.join(save_dir, old)
            try:
                shutil.rmtree(path)
            except Exception:
                pass


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def build_ldlm_components(args):
    """
    Build LDLM autoencoder, diffusion head, and adaptive sampler.
    
    Loads the frozen Qwen3.6 encoder, creates the Perceiver-based
    latent encoder/decoder, and initializes the diffusion head.
    """
    ldlm_cfg = args.model.ldlm or {}

    logger.info_rank0(f"Building LDLM autoencoder (encoder={ldlm_cfg.get('encoder_model_name', args.model.model_path)})")

    autoencoder = LDLMAutoencoder(
        encoder_model_name=ldlm_cfg.get("encoder_model_name", args.model.model_path),
        seq_len=ldlm_cfg.get("seq_len", 128),
        depth=ldlm_cfg.get("depth", 6),
        decoder_input_noise_std=ldlm_cfg.get("decoder_noise_std", 3.0),
        encoder_hidden_layer=ldlm_cfg.get("encoder_hidden_layer", -3),
        decoder_num_layers=ldlm_cfg.get("decoder_num_layers", 3),
        perceiver_heads=ldlm_cfg.get("perceiver_heads", 8),
        share_lm_head=ldlm_cfg.get("share_lm_head", False),
    )

    diffusion_head = DiffusionHead(
        dim=ldlm_cfg.get("diffusion_head_dim") or autoencoder.dim,
        depth=ldlm_cfg.get("diffusion_head_depth", 12),
        heads=ldlm_cfg.get("diffusion_head_heads", 8),
        max_seq_len=ldlm_cfg.get("seq_len", 128),
        cross_attn=ldlm_cfg.get("cross_attn", False),
    )

    sampler = AdaptiveTimestepSampler(
        num_bins=ldlm_cfg.get("adaptive_sampler_num_bins", 100),
        ema_decay=ldlm_cfg.get("adaptive_sampler_ema_decay", 0.999),
        update_interval=ldlm_cfg.get("adaptive_sampler_update_interval", 5000),
    )

    return autoencoder, diffusion_head, sampler


def ldlm_forward(autoencoder, diffusion_head, sampler, input_ids, attention_mask, step, warmup_steps, ldlm_cfg):
    """
    LDLM forward pass with losses.

    Unpacks the rmpad-packed [1, N] tensor into real batches. When cross_attn
    is enabled, each sample is split into a prefix half (cross-attention
    context) and a target half (what gets encoded/denoised/reconstructed).
    """
    use_cross_attn = ldlm_cfg.get("cross_attn", False)
    seq_len = ldlm_cfg.get("seq_len", 64)

    # Unpack: reshape [1, N] -> [B, sample_len] to create real batch dim.
    # When cross_attn is on, each sample needs 2*seq_len tokens (prefix + target).
    sample_len = seq_len * 2 if use_cross_attn else seq_len
    B_in, T_in = input_ids.shape
    if B_in == 1 and T_in > seq_len:
        if T_in < sample_len * 2:
            sample_len = seq_len
        n_samples = T_in // sample_len
        input_ids = input_ids[0, :n_samples * sample_len].reshape(n_samples, sample_len)
        if attention_mask is not None:
            attention_mask = attention_mask[0, :n_samples * sample_len].reshape(n_samples, sample_len)

    B = input_ids.shape[0]

    # Frozen encoder → normalized hidden states for full sample
    h_full = autoencoder.encode_hidden(input_ids, attention_mask)

    # Split prefix / target
    prefix_context = None
    prefix_ids = None
    if use_cross_attn and h_full.shape[1] > seq_len:
        prefix_context = h_full[:, :seq_len, :].detach()
        prefix_ids = input_ids[:, :seq_len]
        h_target = h_full[:, seq_len:, :]
        target_ids = input_ids[:, seq_len:]
    else:
        h_target = h_full
        target_ids = input_ids

    # Perceiver encode → z0 (target only)
    z0 = autoencoder.encode_latent(h_target)

    # Decode z0 back to token predictions
    dec_out = autoencoder.decode(z0, training=True)
    h_hat = dec_out["h_hat"]
    logits = dec_out["logits"]

    # Reconstruction losses (target tokens only)
    min_len = min(h_hat.shape[1], h_target.shape[1], target_ids.shape[1])
    L_h = F.mse_loss(h_hat[:, :min_len], h_target[:, :min_len])
    L_w = F.cross_entropy(
        logits[:, :min_len].reshape(-1, logits.size(-1)),
        target_ids[:, :min_len].reshape(-1),
        ignore_index=-100,
    )

    # Diffusion loss
    skip_diffusion = ldlm_cfg.get("skip_diffusion", False)
    if skip_diffusion:
        L_diff = torch.tensor(0.0, device=z0.device)
        diff_weight = 0.0
    else:
        device = z0.device
        t = sampler.sample(B).to(device)

        d_schedule = ldlm_cfg.get("tangent_d", 3.0)
        alpha_bar = (1.0 - t[:, None, None] ** d_schedule).clamp(min=1e-8)
        sigma = (1.0 - alpha_bar).sqrt().clamp(min=1e-8)

        gamma_min = ldlm_cfg.get("warmup_gamma_min", 0.001)
        if step >= warmup_steps:
            diff_weight = 1.0
        elif warmup_steps <= 0:
            diff_weight = 1.0
        else:
            k = 10.0
            c = 0.8
            s_ratio = step / warmup_steps
            sigma_tilde_s = torch.sigmoid(torch.tensor(k * (s_ratio - c)))
            sigma_tilde_0 = torch.sigmoid(torch.tensor(k * (0.0 - c)))
            sigma_tilde_S = torch.sigmoid(torch.tensor(k * (1.0 - c)))
            diff_weight = (gamma_min + (1.0 - gamma_min) * (sigma_tilde_s - sigma_tilde_0) / (sigma_tilde_S - sigma_tilde_0)).item()

        z0_detached = z0.detach()
        z0_for_diff = z0_detached + diff_weight * (z0 - z0_detached)

        diff_noise = torch.randn_like(z0_for_diff)
        z_t = alpha_bar.sqrt() * z0_for_diff + sigma * diff_noise

        z_hat_prev = None
        sc_prob = ldlm_cfg.get("self_condition_prob", 0.5)
        if torch.rand(1).item() < sc_prob:
            with torch.no_grad():
                z_hat_prev = diffusion_head(z_t, t, context=prefix_context)
                z_hat_prev = z_hat_prev.detach()

        pred = diffusion_head(z_t, t, z_hat_prev=z_hat_prev, context=prefix_context)
        L_diff_per_sample = F.mse_loss(pred, z0_for_diff, reduction='none').mean(dim=(1, 2))
        L_diff = L_diff_per_sample.mean()

        sampler.update(t, L_diff_per_sample.detach())

    recon_h_wt = ldlm_cfg.get("recon_h_weight", 1.0)
    recon_token_wt = ldlm_cfg.get("recon_token_weight", 1.0)
    loss = L_diff + L_h * recon_h_wt + L_w * recon_token_wt

    return {
        "loss": loss,
        "loss_diff": L_diff,
        "loss_recon_h": L_h,
        "loss_recon_token": L_w,
        "diff_weight": diff_weight,
        "t_mean": t.mean() if not skip_diffusion else torch.tensor(0.0),
        "z0": z0,
        "logits": logits,
        "prefix_context": prefix_context,
        "prefix_ids": prefix_ids,
    }


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=45))
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(
        dist_backend=args.train.data_parallel_mode,
        ckpt_manager=args.train.ckpt_manager,
    )

    time_checkpoint_dir = os.path.join(args.train.output_dir, "last_checkpoint")
    time_checkpoint_dir_exists = args.train.save_time_interval_minutes > 0
    if time_checkpoint_dir_exists and args.train.global_rank == 0:
        os.makedirs(time_checkpoint_dir, exist_ok=True)
    if time_checkpoint_dir_exists and dist.get_world_size() > 1:
        dist.barrier()

    latest_checkpoint_path = None
    if args.train.auto_resume:
        if time_checkpoint_dir_exists:
            latest_checkpoint_path = helper.find_latest_time_checkpoint(time_checkpoint_dir)
        if latest_checkpoint_path is None:
            latest_checkpoint_path = helper.find_latest_step_checkpoint(args.train.save_checkpoint_path)
    if args.train.load_checkpoint_path:
        latest_checkpoint_path = args.train.load_checkpoint_path

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    # ------------------------------------------------------------------
    # 1. Tokenizer + Data
    # ------------------------------------------------------------------
    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})
    print(f"tokenizer.mask_token_id: {tokenizer.mask_token_id}")

    if args.data.data_type == "plaintext":
        transform = partial(
            process_pretrain_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "conversation":
        chat_template = build_chat_template(args.data.chat_template, tokenizer)
        transform = partial(
            process_sft_example,
            chat_template=chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "preload":
            import numpy as np
            import pyarrow.parquet as pq
            from torch.utils.data import Dataset as _Dataset

            class _MmapDataset(_Dataset):
                @staticmethod
                def _ensure_mmap(path, seq_len):
                    """Convert parquet to mmap if needed. Returns mmap_dir."""
                    import pyarrow.compute as pc
                    base = path.rsplit(".", 1)[0]
                    mmap_dir = base + ".mmap"
                    ids_path = os.path.join(mmap_dir, "input_ids.npy")
                    if not os.path.exists(ids_path):
                        parquet_path = base + ".tokenized.parquet"
                        if not os.path.exists(parquet_path):
                            raise FileNotFoundError(
                                f"Neither mmap nor parquet found for {path}. Run tasks/build_parquet.py first.")
                        logger.info_rank0(f"Converting parquet -> mmap: {mmap_dir}")
                        os.makedirs(mmap_dir, exist_ok=True)
                        pf = pq.ParquetFile(parquet_path)
                        n_rows = pf.metadata.num_rows
                        mask_path = os.path.join(mmap_dir, "attention_mask.npy")
                        ids_mm = np.memmap(ids_path, dtype=np.int32, mode="w+", shape=(n_rows, seq_len))
                        mask_mm = np.memmap(mask_path, dtype=np.int32, mode="w+", shape=(n_rows, seq_len))
                        row_idx = 0
                        for batch in pf.iter_batches(batch_size=1000000, columns=["input_ids", "attention_mask"]):
                            ids_col = batch.column("input_ids")
                            mask_col = batch.column("attention_mask")
                            lengths = pc.list_value_length(ids_col)
                            valid_mask = pc.equal(lengths, seq_len)
                            ids_flat = pc.list_flatten(pc.filter(ids_col, valid_mask)).to_numpy().astype(np.int32)
                            mask_flat = pc.list_flatten(pc.filter(mask_col, valid_mask)).to_numpy().astype(np.int32)
                            n_valid = len(ids_flat) // seq_len
                            ids_mm[row_idx:row_idx + n_valid] = ids_flat.reshape(n_valid, seq_len)
                            mask_mm[row_idx:row_idx + n_valid] = mask_flat.reshape(n_valid, seq_len)
                            row_idx += n_valid
                            logger.info_rank0(f"  Converted {row_idx:,} / {n_rows:,} rows")
                        ids_mm.flush()
                        mask_mm.flush()
                        np.save(os.path.join(mmap_dir, "meta.npy"), np.array([row_idx, seq_len]))
                        logger.info_rank0(f"Mmap ready: {row_idx:,} examples, {seq_len} seq_len")
                    return mmap_dir

                def __init__(self, paths, seq_len):
                    path_list = [p.strip() for p in paths.split(",")]
                    all_ids = []
                    all_masks = []
                    total = 0
                    for path in path_list:
                        mmap_dir = self._ensure_mmap(path, seq_len)
                        meta = np.load(os.path.join(mmap_dir, "meta.npy"))
                        n_rows, sl = int(meta[0]), int(meta[1])
                        ids_path = os.path.join(mmap_dir, "input_ids.npy")
                        mask_path = os.path.join(mmap_dir, "attention_mask.npy")
                        all_ids.append(np.memmap(ids_path, dtype=np.int32, mode="r", shape=(n_rows, sl)))
                        all_masks.append(np.memmap(mask_path, dtype=np.int32, mode="r", shape=(n_rows, sl)))
                        total += n_rows
                        logger.info_rank0(f"  Loaded {path}: {n_rows:,} examples")
                    self._sources = list(zip(all_ids, all_masks))
                    self._boundaries = []
                    running = 0
                    for ids, _ in self._sources:
                        running += len(ids)
                        self._boundaries.append(running)
                    self._len = total
                    # Pre-shuffle: create a random permutation so web+code are interleaved
                    if len(path_list) > 1:
                        logger.info_rank0(f"Shuffling {total:,} indices across {len(path_list)} sources...")
                        self._perm = np.random.default_rng(42).permutation(total).astype(np.int64)
                        logger.info_rank0(f"Shuffle complete")
                    else:
                        self._perm = None
                    logger.info_rank0(f"Mmap dataset total: {total:,} examples, ~0 MB resident")

                def __len__(self):
                    return self._len

                def _resolve(self, idx):
                    for i, boundary in enumerate(self._boundaries):
                        if idx < boundary:
                            local_idx = idx - (self._boundaries[i - 1] if i > 0 else 0)
                            return self._sources[i], local_idx
                    raise IndexError(f"Index {idx} out of range")

                def __getitem__(self, idx):
                    if self._perm is not None:
                        idx = int(self._perm[idx])
                    (ids, masks), local_idx = self._resolve(idx)
                    return {
                        "input_ids": torch.from_numpy(ids[local_idx].copy()).long(),
                        "attention_mask": torch.from_numpy(masks[local_idx].copy()).long(),
                    }

            train_dataset = _MmapDataset(args.data.train_path, args.data.max_seq_len)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))
        elif args.data.datasets_type == "iterable":
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
        elif args.data.datasets_type == "mapping":
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            enable_masking=args.train.enable_masking,
            mask_token_id=tokenizer.mask_token_id,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    # ------------------------------------------------------------------
    # 2. Build LDLM Components
    # ------------------------------------------------------------------
    logger.info_rank0("Build LDLM components")
    time.sleep(args.train.global_rank * 2)

    autoencoder, diffusion_head, sampler = build_ldlm_components(args)

    # GPU placement strategy
    ldlm_cfg = args.model.ldlm or {}
    n_gpus = torch.cuda.device_count()
    encoder_on_gpu = ldlm_cfg.get("encoder_on_gpu", False)
    if n_gpus >= 2:
        logger.info_rank0(f"Multi-GPU: encoder on cuda:0, trainable on cuda:1")
        autoencoder.move_encoder_to_gpus(max_memory={0: "30GiB", "cpu": "60GiB"})
        trainable_device = torch.device("cuda:1")
    elif encoder_on_gpu:
        logger.info_rank0("Single GPU: encoder AND trainable on cuda:0 (tight VRAM)")
        autoencoder.move_encoder_to_gpus(max_memory={0: "8GiB", "cpu": "60GiB"})
        trainable_device = torch.device("cuda:0")
    else:
        logger.info_rank0("Single GPU: encoder stays on CPU, trainable on cuda:0")
        trainable_device = torch.device("cuda:0")

    # Move sampler to trainable device
    sampler.device = trainable_device
    sampler.bin_edges = sampler.bin_edges.to(trainable_device)
    sampler.loss_ema = sampler.loss_ema.to(trainable_device)
    sampler.bin_probs = sampler.bin_probs.to(trainable_device)

    autoencoder.latent_encoder = autoencoder.latent_encoder.to(trainable_device).to(torch.bfloat16)
    autoencoder.latent_decoder = autoencoder.latent_decoder.to(trainable_device).to(torch.bfloat16)
    if autoencoder.token_decoder is not None:
        autoencoder.token_decoder = autoencoder.token_decoder.to(trainable_device).to(torch.bfloat16)
    autoencoder.lm_head = autoencoder.lm_head.to(trainable_device)
    if hasattr(autoencoder, "final_norm") and autoencoder.final_norm is not None:
        autoencoder.final_norm = autoencoder.final_norm.to(trainable_device)
    diffusion_head = diffusion_head.to(trainable_device).to(torch.bfloat16)

    vocab_size = autoencoder._vocab_size
    model_config = autoencoder.token_encoder.config

    # ------------------------------------------------------------------
    # 3. Optimizer + LR Scheduler (trainable params only)
    # ------------------------------------------------------------------
    # Only train diffusion head + latent encoder/decoder + decoder head
    trainable_params = (
        list(diffusion_head.parameters())
        + list(autoencoder.latent_encoder.parameters())
        + list(autoencoder.latent_decoder.parameters())
    )
    if autoencoder.token_decoder is not None:
        trainable_params += list(autoencoder.token_decoder.parameters())
    if not autoencoder.share_lm_head:
        trainable_params += list(autoencoder.lm_head.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
    )

    from torch.optim.lr_scheduler import ConstantLR
    lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)

    helper.print_device_mem_info("VRAM after building model")

    # ------------------------------------------------------------------
    # 4. Skip FSDP — frozen encoder is already sharded via device_map="auto"
    #    Trainable components (latent encoder/decoder, diffusion head) fit on GPU0.
    # ------------------------------------------------------------------
    full_model = torch.nn.ModuleDict({
        "autoencoder": autoencoder,
        "diffusion_head": diffusion_head,
    })
    model = full_model  # no FSDP wrapping
    model.train()

    # ------------------------------------------------------------------
    # 5. Init wandb
    # ------------------------------------------------------------------
    if args.train.global_rank == 0 and args.train.use_wandb:
        wandb.init(
            project=args.train.wandb_project,
            name=args.train.wandb_name,
            tags=["ldlm"],
            resume="allow",
            entity=args.train.wandb_entity,
            id=args.train.wandb_name,
            config={**vars(args.model), **vars(args.data), **vars(args.train)},
        )

    model_assets = [model_config, tokenizer]
    save_model_assets(args.train.model_assets_dir, model_assets)

    # ------------------------------------------------------------------
    # 6. Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch, start_step, global_step = 0, 0, 0
    use_cross_attn = ldlm_cfg.get("cross_attn", False)
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    time_checkpoint_timer = None
    if time_checkpoint_dir_exists:
        time_checkpoint_timer = helper.PeriodicTimer(args.train.save_time_interval_minutes * 60)
        time_checkpoint_timer.reset()

    if latest_checkpoint_path:
        state = {"model": full_model, "optimizer": optimizer, "extra_state": {}}
        Checkpointer.load(latest_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:
            iter(train_dataloader)

        dist.barrier()
        logger.info_rank0(f"Loaded checkpoint from {latest_checkpoint_path}")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload,
        args.train.enable_gradient_checkpointing,
        args.train.activation_gpu_limit,
    )
    model.train()

    ldlm_cfg = args.model.ldlm or {}

    # ------------------------------------------------------------------
    # 7. Training Loop
    # ------------------------------------------------------------------
    logger.info_rank0(
        f"Start LDLM training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)

        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            step_loss_components: Dict[str, float] = {}

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            torch.cuda.synchronize()
            start_time = time.time()
            last_micro_batch = None

            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)

                micro_batch = {
                    k: v.to(trainable_device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                last_micro_batch = micro_batch

                with model_fwd_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    fwd_out = ldlm_forward(
                        autoencoder=model["autoencoder"],
                        diffusion_head=model["diffusion_head"],
                        sampler=sampler,
                        input_ids=micro_batch["input_ids"],
                        attention_mask=micro_batch.get("attention_mask"),
                        step=global_step,
                        warmup_steps=ldlm_cfg.get("warmup_steps", 50000),
                        ldlm_cfg=ldlm_cfg,
                    )

                    loss_tensor = fwd_out["loss"].mean() / len(micro_batches)
                    step_loss_components["loss_diff"] = (
                        step_loss_components.get("loss_diff", 0.0)
                        + fwd_out["loss_diff"].mean().item() / len(micro_batches)
                    )
                    step_loss_components["loss_recon_h"] = (
                        step_loss_components.get("loss_recon_h", 0.0)
                        + fwd_out["loss_recon_h"].mean().item() / len(micro_batches)
                    )
                    step_loss_components["loss_recon_token"] = (
                        step_loss_components.get("loss_recon_token", 0.0)
                        + fwd_out["loss_recon_token"].mean().item() / len(micro_batches)
                    )

                with model_bwd_context:
                    loss_tensor.backward()

                total_loss += loss_tensor.item()
                del micro_batch

            # Gradient clipping
            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = full_model.clip_grad_norm_(args.train.max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    full_model.parameters(), args.train.max_grad_norm, foreach=True
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # All-reduce losses
            total_loss, grad_norm = all_reduce(
                (total_loss, grad_norm), group=get_parallel_state().fsdp_group
            )
            if step_loss_components:
                names = sorted(step_loss_components.keys())
                values = tuple(step_loss_components[name] for name in names)
                reduced_values = all_reduce(values, group=get_parallel_state().fsdp_group)
                step_loss_components = {name: value for name, value in zip(names, reduced_values)}

            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)
            for name, value in step_loss_components.items():
                train_metrics[f"losses/{name}"] = value

            component_parts = [
                f"{name}:{step_loss_components[name]:.2f}"
                for name in sorted(step_loss_components.keys())
            ]
            postfix_components = ", " + ", ".join(component_parts) if component_parts else ""
            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}{postfix_components}"
            )
            data_loader_tqdm.update()

            # Wandb logging
            if args.train.global_rank == 0 and args.train.use_wandb:
                log_dict = {
                    "training/loss": total_loss,
                    "training/grad_norm": grad_norm,
                    "training/lr": lr,
                    "training/diff_weight": fwd_out["diff_weight"],
                    **{
                        f"training/{k}": v
                        for k, v in step_loss_components.items()
                    },
                }

                # Log latent stats per log_interval
                if global_step % ldlm_cfg.get("log_interval", 50) == 0:
                    log_dict["training/latent_norm"] = fwd_out.get("z0", torch.zeros(1)).norm(dim=-1).mean().item()
                    log_dict["training/timestep_mean"] = fwd_out.get("t_mean", torch.zeros(1)).item()
                    log_dict["training/sampler_loss_min"] = sampler.loss_ema.min().item()
                    log_dict["training/sampler_loss_max"] = sampler.loss_ema.max().item()
                    log_dict["training/sampler_loss_range"] = (
                        sampler.loss_ema.max() - sampler.loss_ema.min()
                    ).item()

                if global_step % 500 == 0:
                    z0_val = fwd_out.get("z0")
                    if z0_val is not None and z0_val.dim() == 3:
                        z0v = z0_val.detach().float()
                        batch_mean = z0v.mean(dim=(0, 1))
                        inter_var = ((z0v.mean(dim=1) - batch_mean) ** 2).mean().item()
                        intra_var = z0v.var(dim=1).mean().item()
                        total_var = z0v.var().item()
                        cos_sim = torch.nn.functional.cosine_similarity(
                            z0v[:, :1, :].expand_as(z0v), z0v, dim=-1
                        ).mean().item()
                        ema_std = full_model.ldlm_autoencoder._z0_ema_std.item()
                        logger.info(
                            f"[Latent diag step={global_step}] "
                            f"total_var={total_var:.4f} inter_sample={inter_var:.4f} "
                            f"intra_seq={intra_var:.4f} cos_sim={cos_sim:.4f} "
                            f"z0_ema_std={ema_std:.4f}"
                        )

                # Log latent histograms per 10x log_interval
                hist_interval = ldlm_cfg.get("log_interval", 50) * 10
                if global_step % hist_interval == 0:
                    z0_val = fwd_out.get("z0")
                    if z0_val is not None:
                        log_dict["latent_stats/z0_mean"] = wandb.Histogram(
                            z0_val.mean(dim=[0, 1]).detach().cpu().float()
                        )
                        log_dict["latent_stats/z0_std"] = wandb.Histogram(
                            z0_val.std(dim=[0, 1]).detach().cpu().float()
                        )

                # Log reconstruction text samples
                text_interval = ldlm_cfg.get("log_interval", 50) * 5
                if global_step % text_interval == 0:
                    try:
                        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = fwd_out.get("logits")
                            if logits is not None:
                                pred_ids = logits.argmax(dim=-1)
                                target_ids = micro_batch.get("input_ids")
                                if target_ids is not None:
                                    min_t = min(pred_ids.shape[1], target_ids.shape[1])
                                    pred_text = tokenizer.decode(pred_ids[0, :min_t], skip_special_tokens=True)
                                    target_text = tokenizer.decode(target_ids[0, :min_t], skip_special_tokens=True)
                                    match_pct = (pred_ids[0, :min_t] == target_ids[0, :min_t]).float().mean().item() * 100
                                    log_dict["samples/target"] = wandb.Html(
                                        f"<b>Target:</b> {target_text}<br><b>Predicted:</b> {pred_text}<br><b>Token match:</b> {match_pct:.1f}%",
                                        inject=False,
                                    )
                    except Exception:
                        pass

                wandb.log(log_dict, step=global_step)

            # Save checkpoint
            save_step = args.train.save_steps and global_step % args.train.save_steps == 0
            eval_step = args.train.eval_every > 0 and global_step % args.train.eval_every == 0
            save_time = False
            if time_checkpoint_dir_exists and time_checkpoint_timer is not None:
                if args.train.global_rank == 0:
                    save_time = time_checkpoint_timer.should_trigger()
                save_time_tensor = torch.tensor([int(save_time)], dtype=torch.int32, device="cuda")
                dist.broadcast(save_time_tensor, src=0)
                save_time = bool(save_time_tensor.item())

            if save_step or eval_step:
                helper.empty_cache()
                if save_step:
                    ckpt_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                elif eval_step:
                    ckpt_path = os.path.join(args.train.save_checkpoint_path, "eval")
                else:
                    raise ValueError("Invalid save or eval step")

                state = {
                    "model": full_model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(ckpt_path, state)
                logger.info_rank0(f"Checkpoint saved to {ckpt_path}")
                prune_checkpoints(args.train.save_checkpoint_path)
                dist.barrier()

                if args.train.global_rank == 0 and args.train.save_hf_weights:
                    hf_path = os.path.join(ckpt_path, "hf_ckpt")
                    state_dict = {}
                    for name, param in full_model.named_parameters():
                        if param.requires_grad:
                            state_dict[name] = param.data
                    for name, buf in full_model.named_buffers():
                        state_dict[name] = buf
                    save_model_weights(hf_path, state_dict, model_assets=model_assets)
                    logger.info_rank0(f"HF weights saved at {hf_path}")

            # Generation eval
            gen_eval_interval = ldlm_cfg.get("gen_eval_interval", 0)
            if gen_eval_interval > 0 and global_step % gen_eval_interval == 0 and args.train.global_rank == 0:
                try:
                    from veomni.models.ldlm.generate import generate, LDLMGenerationConfig
                    gen_config = LDLMGenerationConfig(
                        seq_len=ldlm_cfg.get("seq_len", 64),
                        dim=model["autoencoder"].dim,
                        steps=100,
                        temperature=1.0,
                        self_condition=True,
                        batch_size=4,
                    )
                    model["autoencoder"].eval()
                    model["diffusion_head"].eval()

                    gen_context = None
                    gen_prefix_text = ""
                    if use_cross_attn:
                        gen_context = fwd_out.get("prefix_context")
                        prefix_ids = fwd_out.get("prefix_ids")
                        if gen_context is not None:
                            gen_context = gen_context[:1].expand(4, -1, -1).detach()
                            if prefix_ids is not None:
                                gen_prefix_text = tokenizer.decode(prefix_ids[0], skip_special_tokens=True)[:80]

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        result = generate(
                            diffusion_head=model["diffusion_head"],
                            autoencoder=model["autoencoder"],
                            tokenizer=tokenizer,
                            config=gen_config,
                            device=trainable_device,
                            context=gen_context,
                        )
                    logger.info_rank0(f"=== Generation eval (step {global_step}) ===")
                    if gen_prefix_text:
                        logger.info_rank0(f"  [Prefix] {gen_prefix_text}...")
                    for i, text in enumerate(result["text"]):
                        logger.info_rank0(f"  [Sample {i+1}] {text}")
                    model["autoencoder"].train()
                    model["diffusion_head"].train()
                except Exception as e:
                    logger.info_rank0(f"Gen eval failed: {e}")

            if save_time:
                helper.empty_cache()
                state = {
                    "model": full_model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                helper.save_time_checkpoint(Checkpointer, time_checkpoint_dir, state)
                dist.barrier()
                logger.info_rank0("Time checkpoint refreshed")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM after epoch {epoch + 1}")

        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            ckpt_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": full_model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(ckpt_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Epoch checkpoint saved at {ckpt_path}")
            prune_checkpoints(args.train.save_checkpoint_path)

            if args.train.global_rank == 0 and args.train.save_hf_weights:
                hf_path = os.path.join(ckpt_path, "hf_ckpt")
                state_dict = {}
                for name, param in full_model.named_parameters():
                    if param.requires_grad:
                        state_dict[name] = param.data
                for name, buf in full_model.named_buffers():
                    state_dict[name] = buf
                save_model_weights(hf_path, state_dict, model_assets=model_assets)
                logger.info_rank0(f"HF weights saved at {hf_path}")

    # Final save
    torch.cuda.synchronize()
    del optimizer, lr_scheduler
    helper.empty_cache()

    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        state_dict = {}
        for name, param in full_model.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data
        for name, buf in full_model.named_buffers():
            state_dict[name] = buf
        save_model_weights(hf_path, state_dict, model_assets=model_assets)
        logger.info_rank0(f"Final HF weights saved at {hf_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
