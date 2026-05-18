"""
LDLM text generation script.

Loads trained LDLM weights (diffusion head + Perceiver decoder + token decoder + LM head)
and generates text via iterative latent denoising. The frozen encoder is NOT needed.

Usage:
    python tasks/generate_ldlm.py --checkpoint_path Qwen2.5-3B_LDLM/checkpoints/global_step_500/hf_ckpt
    python tasks/generate_ldlm.py --checkpoint_path Qwen2.5-3B_LDLM/checkpoints/global_step_500/hf_ckpt --steps 20 --batch_size 4
"""

import argparse
import time
import torch
from transformers import AutoTokenizer

from veomni.models.ldlm import LDLMAutoencoder, DiffusionHead
from veomni.models.ldlm.generate import generate, LDLMGenerationConfig


def load_ldlm_for_inference(
    checkpoint_path: str,
    encoder_model_name: str = "Qwen/Qwen2.5-3B",
    seq_len: int = 64,
    depth: int = 4,
    decoder_num_layers: int = 2,
    diffusion_head_depth: int = 3,
    device: str = "cuda",
    share_lm_head: bool = False,
    encoder_hidden_layer: int = -3,
):
    """
    Load LDLM components for inference. The frozen encoder is created
    but immediately discarded — only the trainable components are kept.
    """
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

    diffusion_head = DiffusionHead(
        dim=dim,
        depth=diffusion_head_depth,
    )

    # Load trained weights
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    ae_state = {}
    dh_state = {}
    for k, v in state_dict.items():
        clean_key = k.replace("autoencoder.", "").replace("diffusion_head.", "")
        if k.startswith("autoencoder."):
            ae_state[clean_key] = v
        elif k.startswith("diffusion_head."):
            dh_state[clean_key] = v
        else:
            ae_state[k] = v

    if ae_state:
        autoencoder.load_state_dict(ae_state, strict=False)
    if dh_state:
        diffusion_head.load_state_dict(dh_state, strict=False)

    # Delete the frozen encoder — not needed for generation
    del autoencoder.token_encoder
    autoencoder.token_encoder = None
    torch.cuda.empty_cache()

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    autoencoder = autoencoder.to(device).to(dtype).eval()
    diffusion_head = diffusion_head.to(device).to(dtype).eval()

    return autoencoder, diffusion_head, dim


def main():
    parser = argparse.ArgumentParser(description="LDLM text generation")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained LDLM checkpoint (hf_ckpt directory or .pt file)")
    parser.add_argument("--encoder_model_name", type=str, default="Qwen/Qwen2.5-3B",
                        help="Encoder model name (for tokenizer and architecture config)")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4, help="Perceiver depth")
    parser.add_argument("--decoder_num_layers", type=int, default=2)
    parser.add_argument("--diffusion_head_depth", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of generation rounds")
    parser.add_argument("--no_self_condition", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.encoder_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name, trust_remote_code=True)

    print(f"Loading LDLM checkpoint from {args.checkpoint_path}...")
    autoencoder, diffusion_head, dim = load_ldlm_for_inference(
        checkpoint_path=args.checkpoint_path,
        encoder_model_name=args.encoder_model_name,
        seq_len=args.seq_len,
        depth=args.depth,
        decoder_num_layers=args.decoder_num_layers,
        diffusion_head_depth=args.diffusion_head_depth,
        device=args.device,
    )

    trainable_params = sum(
        p.numel() for p in
        list(autoencoder.parameters()) + list(diffusion_head.parameters())
    )
    print(f"Loaded {trainable_params / 1e6:.1f}M inference parameters on {args.device}")
    print(f"Config: seq_len={args.seq_len}, dim={dim}, steps={args.steps}, "
          f"sampler={args.sampler}, temperature={args.temperature}")
    print("-" * 60)

    gen_config = LDLMGenerationConfig(
        seq_len=args.seq_len,
        dim=dim,
        steps=args.steps,
        sampler=args.sampler,
        temperature=args.temperature,
        self_condition=not args.no_self_condition,
        batch_size=args.batch_size,
    )

    for round_idx in range(args.num_rounds):
        if args.num_rounds > 1:
            print(f"\n--- Round {round_idx + 1}/{args.num_rounds} ---")

        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = generate(
                diffusion_head=diffusion_head,
                autoencoder=autoencoder,
                tokenizer=tokenizer,
                config=gen_config,
                device=torch.device(args.device),
            )
        elapsed = time.perf_counter() - start
        total_tokens = args.batch_size * args.seq_len
        tok_per_sec = total_tokens / elapsed

        for i, text in enumerate(result["text"]):
            print(f"\n[Sample {i + 1}]\n{text}")

        print(f"\n({total_tokens} tokens in {elapsed:.2f}s = {tok_per_sec:.0f} tok/s, "
              f"{args.steps} diffusion steps)")


if __name__ == "__main__":
    main()
