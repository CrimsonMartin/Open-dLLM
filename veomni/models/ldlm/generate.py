"""
LDLM generation: denoise random latents, then decode to tokens.

Supports DDPM and DDIM sampling with self-conditioning and optional
prefix conditioning via cross-attention context.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class LDLMGenerationConfig:
    seq_len: int = 64
    dim: int = 2048
    steps: int = 10
    tangent_d: float = 3.0
    sampler: str = "ddim"  # "ddpm" or "ddim"
    temperature: float = 1.0
    self_condition: bool = True
    batch_size: int = 1


def tangent_schedule(t: torch.Tensor, d: float = 3.0):
    """Tangent noise schedule: alpha_bar = 1 - t^d."""
    alpha_bar = (1.0 - t.pow(d)).clamp(min=1e-8)
    return alpha_bar


@torch.no_grad()
def generate(
    diffusion_head: nn.Module,
    autoencoder: nn.Module,
    tokenizer,
    config: LDLMGenerationConfig,
    device: torch.device = None,
    context: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Generate text via latent diffusion.

    1. Sample z_T from N(0, 1)
    2. Iteratively denoise z_T -> z_0 using the diffusion head
    3. Decode z_0 to tokens via the autoencoder decoder

    Args:
        diffusion_head: DiffusionHead module
        autoencoder: LDLMAutoencoder module
        tokenizer: tokenizer for decoding
        config: generation config
        device: target device
        context: optional (B, P, dim) prefix hidden states for cross-attention

    Returns dict with 'text', 'token_ids', 'latent_history'.
    """
    if device is None:
        device = next(diffusion_head.parameters()).device

    B = config.batch_size
    z = torch.randn(B, config.seq_len, config.dim, device=device)
    z = z * config.temperature

    if context is not None:
        context = context.to(device=device, dtype=z.dtype)

    timesteps = torch.linspace(1.0, 0.0, config.steps + 1, device=device)
    latent_history = [z.cpu()]
    z_hat_prev = None

    for i in range(config.steps):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]

        t_batch = t_now.expand(B)
        alpha_bar_now = tangent_schedule(t_batch, config.tangent_d)[:, None, None]

        z0_pred = diffusion_head(
            z, t_batch,
            z_hat_prev=z_hat_prev if config.self_condition else None,
            context=context,
        )

        z_hat_prev = z0_pred.detach()

        if t_next == 0.0:
            z = z0_pred
        elif config.sampler == "ddim":
            alpha_bar_next = tangent_schedule(t_next.expand(B), config.tangent_d)[:, None, None]
            sigma_now = (1.0 - alpha_bar_now).sqrt()
            sigma_next = (1.0 - alpha_bar_next).sqrt()
            eps_pred = (z - alpha_bar_now.sqrt() * z0_pred) / sigma_now
            z = alpha_bar_next.sqrt() * z0_pred + sigma_next * eps_pred
        else:  # ddpm
            alpha_bar_next = tangent_schedule(t_next.expand(B), config.tangent_d)[:, None, None]
            sigma_now = (1.0 - alpha_bar_now).sqrt()
            sigma_next = (1.0 - alpha_bar_next).sqrt()
            eps_pred = (z - alpha_bar_now.sqrt() * z0_pred) / sigma_now
            noise = torch.randn_like(z) if i < config.steps - 1 else 0.0
            z = alpha_bar_next.sqrt() * z0_pred + sigma_next * (eps_pred * 0.8 + noise * 0.2)

        latent_history.append(z.cpu())

    dec_out = autoencoder.decode(z, training=False)
    logits = dec_out["logits"]
    token_ids = logits.argmax(dim=-1)

    texts = []
    for i in range(B):
        text = tokenizer.decode(token_ids[i], skip_special_tokens=True)
        texts.append(text)

    return {
        "text": texts,
        "token_ids": token_ids.cpu(),
        "logits": logits.cpu(),
        "latent_history": latent_history,
    }
