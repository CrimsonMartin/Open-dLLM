"""
Speculative decoding with LDLM as the draft model.

The LDLM diffusion model drafts K tokens in parallel, then the target
autoregressive model (Qwen2.5) verifies them in a single forward pass.
Accepted tokens are kept; generation continues from the first rejection.

Usage:
    python tasks/spec_decode_ldlm.py \
        --checkpoint_path Qwen2.5-0.5B_LDLM_v10/checkpoints/global_step_2000/hf_ckpt \
        --target_model /workspace/models/qwen2.5-0.5b \
        --prompt "The quick brown fox"

    python tasks/spec_decode_ldlm.py \
        --checkpoint_path Qwen2.5-0.5B_LDLM_v10/checkpoints/global_step_2000/hf_ckpt \
        --target_model /workspace/models/qwen2.5-0.5b \
        --prompt "The quick brown fox" \
        --max_tokens 256 --draft_steps 10 --temperature 0.0
"""

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from veomni.models.ldlm.autoencoder import LDLMAutoencoder, DiffusionHead
from veomni.models.ldlm.generate import tangent_schedule


@dataclass
class SpecDecodeStats:
    total_tokens: int = 0
    accepted_tokens: int = 0
    draft_rounds: int = 0
    draft_time: float = 0.0
    verify_time: float = 0.0
    total_time: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(self.total_tokens, 1)

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / max(self.total_time, 1e-9)

    @property
    def avg_accepted_per_round(self) -> float:
        return self.accepted_tokens / max(self.draft_rounds, 1)


def load_target_model(model_path: str, device: str = "cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model


def load_ldlm_draft(
    checkpoint_path: str,
    encoder_model_name: str,
    seq_len: int = 64,
    depth: int = 4,
    decoder_num_layers: int = 2,
    diffusion_head_depth: int = 12,
    share_lm_head: bool = True,
    encoder_hidden_layer: int = -1,
    cross_attn: bool = False,
    device: str = "cuda",
):
    autoencoder = LDLMAutoencoder(
        encoder_model_name=encoder_model_name,
        seq_len=seq_len,
        depth=depth,
        decoder_input_noise_std=0.0,
        encoder_hidden_layer=encoder_hidden_layer,
        decoder_num_layers=decoder_num_layers,
        share_lm_head=share_lm_head,
    )
    dim = autoencoder.dim

    diffusion_head = DiffusionHead(dim=dim, depth=diffusion_head_depth, cross_attn=cross_attn)

    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        import glob
        import os
        if os.path.isdir(checkpoint_path):
            safetensor_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
            pt_files = glob.glob(os.path.join(checkpoint_path, "*.pt"))
            if safetensor_files:
                from safetensors.torch import load_file
                state_dict = load_file(safetensor_files[0])
            elif pt_files:
                state_dict = torch.load(pt_files[0], map_location="cpu", weights_only=True)
            else:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if "model" in state_dict:
        state_dict = state_dict["model"]

    ae_state, dh_state, buf_state = {}, {}, {}
    for k, v in state_dict.items():
        if k.startswith("ldlm_autoencoder."):
            clean = k[len("ldlm_autoencoder."):]
            ae_state[clean] = v
        elif k.startswith("ldlm_diffusion_head."):
            clean = k[len("ldlm_diffusion_head."):]
            dh_state[clean] = v
        elif k.startswith("autoencoder."):
            ae_state[k[len("autoencoder."):]] = v
        elif k.startswith("diffusion_head."):
            dh_state[k[len("diffusion_head."):]] = v

    if ae_state:
        autoencoder.load_state_dict(ae_state, strict=False)
    if dh_state:
        diffusion_head.load_state_dict(dh_state, strict=False)

    del autoencoder.token_encoder
    autoencoder.token_encoder = None
    torch.cuda.empty_cache()

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    autoencoder = autoencoder.to(device).to(dtype).eval()
    diffusion_head = diffusion_head.to(device).to(dtype).eval()

    return autoencoder, diffusion_head, dim


@torch.no_grad()
def ldlm_draft(
    diffusion_head,
    autoencoder,
    K: int,
    dim: int,
    steps: int = 10,
    tangent_d: float = 3.0,
    temperature: float = 1.0,
    device: torch.device = None,
    context: Optional[torch.Tensor] = None,
):
    """
    Generate K draft tokens via latent diffusion.

    Args:
        context: optional (1, P, dim) prefix hidden states for cross-attention

    Returns token_ids (1, K) and logits (1, K, vocab).
    """
    z = torch.randn(1, K, dim, device=device) * temperature

    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
    z_hat_prev = None

    for i in range(steps):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]

        t_batch = t_now.unsqueeze(0)
        alpha_bar_now = tangent_schedule(t_batch, tangent_d)[:, None, None]

        z0_pred = diffusion_head(z, t_batch, z_hat_prev=z_hat_prev, context=context)
        z_hat_prev = z0_pred.detach()

        if t_next == 0.0:
            z = z0_pred
        else:
            alpha_bar_next = tangent_schedule(t_next.unsqueeze(0), tangent_d)[:, None, None]
            sigma_now = (1.0 - alpha_bar_now).sqrt()
            sigma_next = (1.0 - alpha_bar_next).sqrt()
            eps_pred = (z - alpha_bar_now.sqrt() * z0_pred) / sigma_now
            z = alpha_bar_next.sqrt() * z0_pred + sigma_next * eps_pred

    dec_out = autoencoder.decode(z, training=False)
    logits = dec_out["logits"]
    token_ids = logits.argmax(dim=-1)

    return token_ids, logits


@torch.no_grad()
def verify_tokens(
    target_model,
    prefix_ids: torch.Tensor,
    draft_ids: torch.Tensor,
    temperature: float = 0.0,
):
    """
    Verify draft tokens against the target model.

    Runs a single forward pass on [prefix + draft] and checks whether the
    target model's greedy (or sampled) predictions match the draft.

    Returns:
        n_accepted: number of draft tokens accepted (0 to K)
        bonus_token: the target model's next token after the last accepted position
    """
    combined = torch.cat([prefix_ids, draft_ids], dim=-1)
    outputs = target_model(input_ids=combined)
    target_logits = outputs.logits

    K = draft_ids.shape[-1]
    prefix_len = prefix_ids.shape[-1]

    n_accepted = 0
    for i in range(K):
        pos = prefix_len - 1 + i
        if temperature == 0.0:
            target_token = target_logits[0, pos].argmax(dim=-1)
        else:
            probs = F.softmax(target_logits[0, pos] / temperature, dim=-1)
            target_token = torch.multinomial(probs, 1).squeeze(-1)

        if target_token.item() == draft_ids[0, i].item():
            n_accepted += 1
        else:
            bonus = target_token.unsqueeze(0).unsqueeze(0)
            return n_accepted, bonus

    last_pos = prefix_len - 1 + K
    if temperature == 0.0:
        bonus_token = target_logits[0, last_pos].argmax(dim=-1)
    else:
        probs = F.softmax(target_logits[0, last_pos] / temperature, dim=-1)
        bonus_token = torch.multinomial(probs, 1).squeeze(-1)
    bonus = bonus_token.unsqueeze(0).unsqueeze(0)

    return n_accepted, bonus


@torch.no_grad()
def speculative_decode(
    target_model,
    diffusion_head,
    autoencoder,
    tokenizer,
    prompt: str,
    max_tokens: int = 128,
    draft_k: int = 64,
    draft_steps: int = 10,
    tangent_d: float = 3.0,
    temperature: float = 0.0,
    draft_temperature: float = 1.0,
    device: torch.device = None,
    verbose: bool = True,
):
    """
    Generate text using speculative decoding.

    The LDLM draft model proposes draft_k tokens at a time,
    then the target AR model verifies them in a single forward pass.
    """
    if device is None:
        device = next(target_model.parameters()).device

    dim = autoencoder.dim
    stats = SpecDecodeStats()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids.clone()
    tokens_generated = 0

    if verbose:
        print(f"Prompt ({input_ids.shape[-1]} tokens): {prompt}")
        print(f"Config: draft_k={draft_k}, draft_steps={draft_steps}, "
              f"max_tokens={max_tokens}, temperature={temperature}")
        print("-" * 60)

    t_start = time.perf_counter()

    use_cross_attn = hasattr(diffusion_head, 'has_cross_attn') and diffusion_head.has_cross_attn

    while tokens_generated < max_tokens:
        remaining = max_tokens - tokens_generated
        k = min(draft_k, remaining)

        # Get prefix hidden states from target model for cross-attention
        prefix_context = None
        if use_cross_attn:
            target_out = target_model(
                input_ids=generated_ids,
                output_hidden_states=True,
            )
            prefix_context = target_out.hidden_states[-1].to(dtype=torch.bfloat16)

        t0 = time.perf_counter()
        draft_ids, draft_logits = ldlm_draft(
            diffusion_head=diffusion_head,
            autoencoder=autoencoder,
            K=k,
            dim=dim,
            steps=draft_steps,
            tangent_d=tangent_d,
            temperature=draft_temperature,
            device=device,
            context=prefix_context,
        )
        stats.draft_time += time.perf_counter() - t0

        t0 = time.perf_counter()
        n_accepted, bonus_token = verify_tokens(
            target_model=target_model,
            prefix_ids=generated_ids,
            draft_ids=draft_ids,
            temperature=temperature,
        )
        stats.verify_time += time.perf_counter() - t0

        if n_accepted > 0:
            generated_ids = torch.cat(
                [generated_ids, draft_ids[:, :n_accepted]], dim=-1
            )
        generated_ids = torch.cat([generated_ids, bonus_token], dim=-1)

        new_tokens = n_accepted + 1
        tokens_generated += new_tokens
        stats.total_tokens += k
        stats.accepted_tokens += n_accepted
        stats.draft_rounds += 1

        if verbose:
            accepted_text = tokenizer.decode(
                draft_ids[0, :n_accepted], skip_special_tokens=True
            ) if n_accepted > 0 else ""
            bonus_text = tokenizer.decode(bonus_token[0], skip_special_tokens=True)
            print(f"  Round {stats.draft_rounds}: "
                  f"drafted {k}, accepted {n_accepted}/{k}, "
                  f"bonus='{bonus_text}'"
                  f"{f', accepted: {repr(accepted_text)}' if n_accepted > 0 else ''}")

        if tokenizer.eos_token_id is not None:
            if bonus_token[0, 0].item() == tokenizer.eos_token_id:
                if verbose:
                    print("  [EOS reached]")
                break

    stats.total_time = time.perf_counter() - t_start

    output_ids = generated_ids[0, input_ids.shape[-1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, generated_ids, stats


@torch.no_grad()
def baseline_ar_generate(
    target_model,
    tokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    device: torch.device = None,
):
    """Baseline autoregressive generation for speed comparison."""
    if device is None:
        device = next(target_model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids.clone()

    t_start = time.perf_counter()
    for _ in range(max_tokens):
        outputs = target_model(input_ids=generated_ids)
        if temperature == 0.0:
            next_token = outputs.logits[0, -1].argmax(dim=-1)
        else:
            probs = F.softmax(outputs.logits[0, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
        generated_ids = torch.cat(
            [generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1
        )
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break
    elapsed = time.perf_counter() - t_start

    output_ids = generated_ids[0, input_ids.shape[-1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, elapsed, len(output_ids)


def main():
    parser = argparse.ArgumentParser(description="Speculative decoding with LDLM draft model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to LDLM checkpoint (hf_ckpt directory)")
    parser.add_argument("--target_model", type=str, required=True,
                        help="Path to target AR model (e.g. Qwen2.5-0.5B)")
    parser.add_argument("--prompt", type=str, default="The quick brown fox",
                        help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--draft_k", type=int, default=64,
                        help="Tokens per draft round (matches LDLM seq_len)")
    parser.add_argument("--draft_steps", type=int, default=10,
                        help="Diffusion denoising steps per draft")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--draft_temperature", type=float, default=1.0,
                        help="Diffusion noise temperature")
    parser.add_argument("--tangent_d", type=float, default=3.0)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--decoder_num_layers", type=int, default=2)
    parser.add_argument("--diffusion_head_depth", type=int, default=12)
    parser.add_argument("--share_lm_head", action="store_true", default=True)
    parser.add_argument("--no_share_lm_head", dest="share_lm_head", action="store_false")
    parser.add_argument("--encoder_hidden_layer", type=int, default=-1)
    parser.add_argument("--cross_attn", action="store_true", default=False,
                        help="Enable cross-attention conditioning on prefix hidden states")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run baseline AR generation for comparison")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading tokenizer from {args.target_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    print(f"Loading target AR model from {args.target_model}...")
    target_model = load_target_model(args.target_model, device=args.device)
    target_params = sum(p.numel() for p in target_model.parameters())
    print(f"  Target model: {target_params / 1e6:.0f}M params")

    print(f"Loading LDLM draft model from {args.checkpoint_path}...")
    autoencoder, diffusion_head, dim = load_ldlm_draft(
        checkpoint_path=args.checkpoint_path,
        encoder_model_name=args.target_model,
        seq_len=args.seq_len,
        depth=args.depth,
        decoder_num_layers=args.decoder_num_layers,
        diffusion_head_depth=args.diffusion_head_depth,
        share_lm_head=args.share_lm_head,
        encoder_hidden_layer=args.encoder_hidden_layer,
        cross_attn=args.cross_attn,
        device=args.device,
    )
    draft_params = sum(
        p.numel() for p in
        list(autoencoder.parameters()) + list(diffusion_head.parameters())
    )
    print(f"  Draft model: {draft_params / 1e6:.1f}M params, dim={dim}")
    print("=" * 60)

    # --- Speculative decoding ---
    print("\n[Speculative Decoding]")
    output_text, generated_ids, stats = speculative_decode(
        target_model=target_model,
        diffusion_head=diffusion_head,
        autoencoder=autoencoder,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        draft_k=args.draft_k,
        draft_steps=args.draft_steps,
        tangent_d=args.tangent_d,
        temperature=args.temperature,
        draft_temperature=args.draft_temperature,
        device=device,
    )

    print(f"\n{'=' * 60}")
    print(f"Output: {output_text}")
    print(f"\nStats:")
    print(f"  Rounds:            {stats.draft_rounds}")
    print(f"  Acceptance rate:   {stats.acceptance_rate:.1%} "
          f"({stats.accepted_tokens}/{stats.total_tokens})")
    print(f"  Avg accepted/rnd:  {stats.avg_accepted_per_round:.1f}")
    print(f"  Draft time:        {stats.draft_time:.3f}s")
    print(f"  Verify time:       {stats.verify_time:.3f}s")
    print(f"  Total time:        {stats.total_time:.3f}s")
    print(f"  Throughput:        {stats.tokens_per_second:.0f} tok/s")

    # --- Baseline AR ---
    if args.baseline:
        print(f"\n{'=' * 60}")
        print("[Baseline Autoregressive]")
        ar_text, ar_time, ar_tokens = baseline_ar_generate(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device,
        )
        ar_tps = ar_tokens / ar_time
        print(f"Output: {ar_text}")
        print(f"\n  Tokens:     {ar_tokens}")
        print(f"  Time:       {ar_time:.3f}s")
        print(f"  Throughput: {ar_tps:.0f} tok/s")

        print(f"\n{'=' * 60}")
        print("[Comparison]")
        spec_tps = stats.tokens_per_second
        speedup = spec_tps / ar_tps if ar_tps > 0 else 0
        print(f"  AR baseline:    {ar_tps:.0f} tok/s")
        print(f"  Spec decode:    {spec_tps:.0f} tok/s")
        print(f"  Speedup:        {speedup:.2f}x")


if __name__ == "__main__":
    main()
