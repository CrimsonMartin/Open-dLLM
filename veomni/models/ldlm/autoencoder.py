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
LDLM (Latent Diffusion Language Model) autoencoder module.

Implements the Perceiver-based latent encoder/decoder that compresses
Qwen3.6 hidden states into a compact latent space for diffusion modeling.

Reference: LDLM (arXiv:2605.07933)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Perceiver components
# ---------------------------------------------------------------------------

class PreNorm(nn.Module):
    """Pre-normalization wrapper for any module."""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, context: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if context is not None:
            return self.fn(self.norm(x), context=context, **kwargs)
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Simple MLP with GELU activation."""
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden_dim = dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """Cross-attention: queries attend to key-value pairs."""
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        H = self.heads

        q = self.to_q(x).reshape(B, N, H, C // H).permute(0, 2, 1, 3)   # (B, H, N, d)
        k = self.to_k(context).reshape(B, -1, H, C // H).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, -1, H, C // H).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class SelfAttention(nn.Module):
    """Self-attention with pre-norm."""
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = self.heads

        qkv = self.to_qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """
    Perceiver-style resampler that maps a variable-length input sequence
    to a fixed set of latent tokens via cross-attention.

    Architecture (from Flamingo / LDLM):
        latents -> self-attention -> cross-attention(latents, context) -> FFN
    """
    def __init__(
        self,
        dim: int,
        num_latents: int = 128,
        depth: int = 6,
        heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, heads=heads)),
                PreNorm(dim, CrossAttention(dim, heads=heads)),
                PreNorm(dim, FeedForward(dim, hidden_mult=ff_mult)),
            ]))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, T, dim) input sequence from encoder
        Returns:
            latents: (B, num_latents, dim) compressed representation
        """
        B = context.shape[0]
        x = self.latents.expand(B, -1, -1)

        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context) + x
            x = ff(x) + x

        return x


# ---------------------------------------------------------------------------
# LDLM Autoencoder
# ---------------------------------------------------------------------------

class LDLMAutoencoder(nn.Module):
    """
    Latent Diffusion Language Model autoencoder.

    Encodes Qwen3.6 hidden states into a compact latent space via a
    Perceiver resampler, then decodes back to token predictions.

    Architecture:
        Qwen3.6 encoder (frozen) -> Perceiver encoder -> latent z
        -> Perceiver decoder -> Transformer decoder -> LM head
    """
    def __init__(
        self,
        encoder_model_name: str = "Qwen/Qwen3.6-35B-A3B",
        seq_len: int = 128,
        depth: int = 6,
        decoder_input_noise_std: float = 3.0,
        latent_dim: Optional[int] = None,
        encoder_hidden_layer: int = -3,
        decoder_num_layers: int = 3,
        perceiver_heads: int = 8,
        skip_encoder: bool = False,
        share_lm_head: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.decoder_input_noise_std = decoder_input_noise_std
        self.encoder_hidden_layer = encoder_hidden_layer
        self.share_lm_head = share_lm_head

        self.register_buffer("_h_mean", torch.zeros(1))
        self.register_buffer("_h_var", torch.ones(1))
        self.register_buffer("_h_count", torch.tensor(0.0))
        self._normalize_hidden_states = True

        self.register_buffer("_z0_ema_std", torch.ones(1))

        if skip_encoder:
            self.token_encoder = None
            self._encoder_device_map = None
            cfg = AutoConfig.from_pretrained(encoder_model_name, trust_remote_code=True)
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
                self.dim = cfg.text_config.hidden_size
                vocab_size = cfg.text_config.vocab_size
            else:
                self.dim = cfg.hidden_size
                vocab_size = cfg.vocab_size
        else:
            self.token_encoder = AutoModel.from_pretrained(
                encoder_model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
            self._encoder_device_map = None
            for p in self.token_encoder.parameters():
                p.requires_grad = False

            cfg = self.token_encoder.config
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
                self.dim = cfg.text_config.hidden_size
                vocab_size = cfg.text_config.vocab_size
            else:
                self.dim = cfg.hidden_size
                vocab_size = cfg.vocab_size
        self.latent_dim = latent_dim or self.dim
        self._vocab_size = vocab_size

        # Compute nhead from dim (must divide evenly)
        for nhead_candidate in [16, 12, 8, 4, 2]:
            if self.dim % nhead_candidate == 0:
                decoder_nhead = nhead_candidate
                break
        else:
            decoder_nhead = 8  # safe fallback

        # Perceiver-based latent encoder/decoder
        self.latent_encoder = PerceiverResampler(
            dim=self.dim,
            num_latents=seq_len,
            depth=depth,
            heads=perceiver_heads,
        )
        self.latent_decoder = PerceiverResampler(
            dim=self.dim,
            num_latents=seq_len,
            depth=depth,
            heads=perceiver_heads,
        )

        if share_lm_head:
            # Use the target model's pre-trained LM head (frozen) with its
            # final RMSNorm, matching the original model's forward path:
            #   last_hidden -> RMSNorm -> lm_head -> logits
            self.token_decoder = None
            if self.token_encoder is not None:
                lm_weight = self.token_encoder.embed_tokens.weight.data.clone().float()
                norm_weight = self._get_final_norm_weight(self.token_encoder)
            else:
                from transformers import AutoModelForCausalLM
                tmp = AutoModelForCausalLM.from_pretrained(
                    encoder_model_name, trust_remote_code=True, torch_dtype=torch.float32,
                )
                lm_weight = tmp.lm_head.weight.data.clone()
                norm_weight = self._get_final_norm_weight(tmp.model if hasattr(tmp, "model") else tmp)
                del tmp
            self.final_norm = nn.RMSNorm(self.dim, eps=1e-6)
            if norm_weight is not None:
                self.final_norm.weight = nn.Parameter(norm_weight, requires_grad=False)
            else:
                self.final_norm.weight.requires_grad = False
            self.lm_head = nn.Linear(self.dim, self._vocab_size, bias=False)
            self.lm_head.weight = nn.Parameter(lm_weight, requires_grad=False)
        else:
            # Original LDLM: learned token decoder + fresh LM head
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.dim, nhead=decoder_nhead, dim_feedforward=self.dim * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=decoder_num_layers,
                norm=nn.LayerNorm(self.dim),
            )
            self.lm_head = nn.Linear(self.dim, self._vocab_size)

    @staticmethod
    def _get_final_norm_weight(model_or_base):
        """Extract the final RMSNorm weight from a HuggingFace model."""
        for attr in ("norm", "final_layernorm", "ln_f"):
            norm = getattr(model_or_base, attr, None)
            if norm is not None and hasattr(norm, "weight"):
                return norm.weight.data.clone().float()
        return None

    def move_encoder_to_gpus(self, max_memory=None):
        import transformers.modeling_utils as mu
        _orig_warmup = mu.caching_allocator_warmup
        mu.caching_allocator_warmup = lambda *a, **k: None
        try:
            from transformers import AutoModel
            self.token_encoder = AutoModel.from_pretrained(
                self.token_encoder.config._name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
            )
            for p in self.token_encoder.parameters():
                p.requires_grad = False
            if hasattr(self.token_encoder, "hf_device_map"):
                self._encoder_device_map = dict(self.token_encoder.hf_device_map)
            else:
                self._encoder_device_map = {"": 0}
        finally:
            mu.caching_allocator_warmup = _orig_warmup

    def encode_hidden(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run frozen encoder and return normalized hidden states (before Perceiver)."""
        device = self.latent_encoder.latents.device
        with torch.no_grad():
            if self._encoder_device_map is not None:
                encoder_device = self.token_encoder.device
                outputs = self.token_encoder(
                    input_ids.to(encoder_device),
                    attention_mask=attention_mask.to(encoder_device) if attention_mask is not None else None,
                    output_hidden_states=True,
                )
                h = outputs.hidden_states[self.encoder_hidden_layer].to(device)
            else:
                outputs = self.token_encoder(
                    input_ids.cpu(),
                    attention_mask=attention_mask.cpu() if attention_mask is not None else None,
                    output_hidden_states=True,
                )
                h = outputs.hidden_states[self.encoder_hidden_layer].to(device)
            del outputs

        if self._normalize_hidden_states and self.training:
            h_mean = self._h_mean.to(h.device)
            h_var = self._h_var.to(h.device)
            h_count = self._h_count.to(h.device)
            batch_mean = h.mean()
            batch_var = h.var(unbiased=False)
            count = h.numel()
            new_count = h_count + count
            h_mean = h_mean * (h_count / new_count) + batch_mean * (count / new_count)
            h_var = h_var * (h_count / new_count) + batch_var * (count / new_count)
            self._h_mean.copy_(h_mean)
            self._h_var.copy_(h_var)
            self._h_count.copy_(new_count)

        if self._normalize_hidden_states and self._h_count > 0:
            h = (h - self._h_mean.to(h.device)) / (self._h_var.to(h.device).sqrt().clamp(min=1e-6))

        return h

    def encode_latent(self, h: torch.Tensor) -> torch.Tensor:
        """Run Perceiver encoder + z0 normalization on hidden states."""
        z0 = self.latent_encoder(h)

        with torch.no_grad():
            z0_std = z0.std().clamp(min=0.01)
            if self.training:
                self._z0_ema_std.mul_(0.99).add_(z0_std.to(self._z0_ema_std.device), alpha=0.01)
        z0 = z0 / self._z0_ema_std.to(z0.device).clamp(min=0.01)

        return z0

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict:
        h = self.encode_hidden(input_ids, attention_mask)
        z0 = self.encode_latent(h)
        return {"z0": z0, "h": h}

    def decode(self, z: torch.Tensor, training: bool = True) -> Dict:
        """
        Decode latent back to token predictions.

        Args:
            z: (B, seq_len, dim) latent representation
            training: if True, add noise to latent before decoding
        Returns:
            dict with 'logits', 'h_hat', 'noise'
        """
        if training and self.decoder_input_noise_std > 0:
            noise = torch.randn_like(z) * self.decoder_input_noise_std
            z_noisy = z + noise
        else:
            z_noisy = z
            noise = None

        h_hat = self.latent_decoder(z_noisy)

        if self.share_lm_head:
            # Denormalize back to original scale for the frozen LM head.
            h_for_lm = h_hat
            if self._h_count > 0:
                h_std = self._h_var.to(h_hat.device).sqrt().clamp(min=1e-6)
                h_for_lm = h_hat * h_std + self._h_mean.to(h_hat.device)
            logits = self.lm_head(self.final_norm(h_for_lm))
        else:
            # Original LDLM: detached token decoder + learned LM head
            h_detached = h_hat.detach() if training else h_hat
            token_hidden = self.token_decoder(
                tgt=h_detached,
                memory=h_detached,
            )
            logits = self.lm_head(token_hidden)

        return {
            "logits": logits,
            "h_hat": h_hat,
            "noise": noise,
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, training: bool = True) -> Dict:
        """
        Full forward pass: encode -> (noise) -> decode.

        Args:
            input_ids: (B, T) token IDs
            attention_mask: optional (B, T) mask
            training: if True, apply decoder input noise
        Returns:
            dict with 'z0', 'h_hat', 'logits', 'h', 'noise'
        """
        enc_out = self.encode(input_ids, attention_mask)
        dec_out = self.decode(enc_out["z0"], training=training)

        return {
            "z0": enc_out["z0"],
            "h": enc_out["h"],
            "h_hat": dec_out["h_hat"],
            "logits": dec_out["logits"],
            "noise": dec_out["noise"],
        }


# ---------------------------------------------------------------------------
# Diffusion Head (DiT with AdaLN)
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for scalar timesteps."""
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half))
    args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    return emb.to(t.dtype) if t.is_floating_point() else emb


class AdaLNDiTBlock(nn.Module):
    """DiT block with Adaptive LayerNorm-Zero conditioning and optional cross-attention."""
    def __init__(self, dim: int, heads: int = 8, has_cross_attn: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, heads=heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = FeedForward(dim)

        self.has_cross_attn = has_cross_attn
        if has_cross_attn:
            self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
            self.cross_attn = CrossAttention(dim, heads=heads)
            self.cross_gate = nn.Parameter(torch.zeros(1, 1, dim))
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        else:
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = \
            self.adaLN(cond).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_attn) + shift_attn
        x = x + gate_attn * self.attn(h)
        if self.has_cross_attn and context is not None:
            x = x + self.cross_gate * self.cross_attn(self.norm_cross(x), context=context)
        h = self.norm2(x) * (1 + scale_ff) + shift_ff
        x = x + gate_ff * self.ff(h)
        return x


class DiffusionHead(nn.Module):
    """
    DiT-style diffusion transformer for latent denoising.

    Supports optional prefix conditioning via cross-attention: when a
    ``context`` tensor (prefix hidden states) is provided, each block
    cross-attends to it so the diffusion head can generate continuations
    rather than unconditional samples.

    - Sinusoidal time embedding (not raw scalar)
    - Learned 1D position embeddings (breaks permutation symmetry)
    - AdaLN-Zero conditioning at every layer (not additive-once)
    - Self-conditioning via proj_in concat
    - Cross-attention to prefix context (optional)
    """
    def __init__(self, dim: int = 2048, depth: int = 12, heads: int = 8,
                 max_seq_len: int = 512, cross_attn: bool = False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.has_cross_attn = cross_attn

        time_inner = min(dim, 1024)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_inner, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_embed_dim = time_inner

        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        self.proj_in = nn.Linear(dim * 2, dim)

        self.blocks = nn.ModuleList([
            AdaLNDiTBlock(dim, heads=heads, has_cross_attn=cross_attn)
            for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                z_hat_prev: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = z.shape
        if t.dim() == 2:
            t = t.squeeze(-1)
        t_emb = sinusoidal_embedding(t, self.time_embed_dim).to(z.dtype)
        cond = self.time_mlp(t_emb)

        if z_hat_prev is None:
            z_hat_prev = torch.zeros_like(z)
        x = self.proj_in(torch.cat([z, z_hat_prev], dim=-1))
        x = x + self.pos_embed[:, :T, :]

        for block in self.blocks:
            x = block(x, cond, context=context)

        shift, scale = self.out_adaLN(cond).unsqueeze(1).chunk(2, dim=-1)
        x = self.out_norm(x) * (1 + scale) + shift
        return x


ModelClass = LDLMAutoencoder