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
Training loop hooks for LDLM.

These patches integrate the LDLM autoencoder and diffusion head into
the Open-dLLM training loop defined in tasks/train_torch.py.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from veomni.models.ldlm.autoencoder import LDLMAutoencoder, DiffusionHead
from veomni.models.ldlm.sampler import AdaptiveTimestepSampler


class LDLMTrainer:
    """
    Wraps an LDLM autoencoder + diffusion head for training.

    Manages:
    - Forward pass through autoencoder and diffusion head
    - Adaptive timestep sampling
    - Loss computation (diffusion + reconstruction)
    - Warmup schedule
    """

    def __init__(
        self,
        autoencoder: LDLMAutoencoder,
        config,
    ):
        self.autoencoder = autoencoder
        self.config = config
        ldlm_cfg = config.get("ldlm", {})

        self.dim = autoencoder.dim
        self.latent_dim = autoencoder.latent_dim

        # Diffusion head
        diff_dim = ldlm_cfg.get("diffusion_head_dim") or self.dim
        diff_depth = ldlm_cfg.get("diffusion_head_depth", 12)
        self.diffusion_head = DiffusionHead(dim=diff_dim, depth=diff_depth)

        # Adaptive timestep sampler
        self.sampler = AdaptiveTimestepSampler(
            num_bins=ldlm_cfg.get("adaptive_sampler_num_bins", 100),
            ema_decay=ldlm_cfg.get("adaptive_sampler_ema_decay", 0.999),
            update_interval=ldlm_cfg.get("adaptive_sampler_update_interval", 5000),
        )

        # Warmup state
        self.warmup_steps = ldlm_cfg.get("warmup_steps", 50000)
        self.global_step = 0

        # Loss weights
        self.recon_h_weight = ldlm_cfg.get("recon_h_weight", 1.0)
        self.recon_token_weight = ldlm_cfg.get("recon_token_weight", 1.0)

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step.

        Args:
            input_ids: (B, T) token IDs
            attention_mask: optional (B, T) mask
        Returns:
            dict of scalar losses
        """
        B = input_ids.shape[0]

        # 1. Autoencoder forward
        ae_out = self.autoencoder(input_ids, attention_mask, training=True)
        z0 = ae_out["z0"]          # clean latent
        h = ae_out["h"]            # encoder hidden state
        h_hat = ae_out["h_hat"]    # decoded hidden state
        logits = ae_out["logits"]  # token logits

        # 2. Reconstruction losses
        L_h = F.mse_loss(h_hat, h)
        L_w = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            ignore_index=-100,
        )

        # 3. Adaptive timestep sampling
        t = self.sampler.sample(B)

        # 4. Diffusion on latents
        # Simple schedule: alpha_bar(t) = 1 - t^2 (cosine schedule)
        alpha_bar = 1.0 - t[:, None, None] ** 2
        sigma = (1.0 - alpha_bar).sqrt()

        noise = torch.randn_like(z0)
        z_t = alpha_bar.sqrt() * z0 + sigma * noise

        pred = self.diffusion_head(z_t, t)

        L_diff = F.mse_loss(pred, z0)

        # 5. Warmup: linearly increase diffusion weight from 0
        warmup_progress = min(self.global_step / max(self.warmup_steps, 1), 1.0)
        diff_weight = warmup_progress

        # 6. Total loss
        loss = L_diff * diff_weight + L_h * self.recon_h_weight + L_w * self.recon_token_weight

        # 7. Update sampler
        self.sampler.update(t, L_diff.detach())

        self.global_step += 1

        return {
            "loss": loss,
            "loss_diff": L_diff,
            "loss_recon_h": L_h,
            "loss_recon_token": L_w,
            "diff_weight": torch.tensor(diff_weight),
            "t_mean": t.mean(),
        }

    @torch.no_grad()
    def generate(
        self,
        z0: torch.Tensor,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """
        Generate tokens from a latent via DDIM reverse process.

        Args:
            z0: (B, T, dim) latent
            num_steps: number of reverse diffusion steps
        Returns:
            token_ids: (B, T) generated tokens
        """
        B, T, D = z0.shape
        z = torch.randn_like(z0)

        for i in range(num_steps):
            t_val = 1.0 - (i / num_steps)
            t = torch.full((B,), t_val, device=z0.device)

            pred_z0 = self.diffusion_head(z, t)

            # DDIM update
            alpha_bar = 1.0 - t_val ** 2
            z = alpha_bar.sqrt() * pred_z0 + (1.0 - alpha_bar).sqrt() * torch.randn_like(z)

        # Decode latent to tokens
        dec_out = self.autoencoder.decode(z, training=False)
        tokens = dec_out["logits"].argmax(dim=-1)
        return tokens

    def get_sampler_stats(self) -> Dict:
        """Return sampler statistics for logging."""
        return {
            "sampler_loss_min": self.sampler.loss_ema.min().item(),
            "sampler_loss_max": self.sampler.loss_ema.max().item(),
            "sampler_update_count": self.sampler.update_counter,
        }
