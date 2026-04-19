#!/usr/bin/env python3
"""
Test GRPO training with MegaTrain backend on Qwen3.5-27B (single GPU).
Uses REAL GSM8K prompts tokenized by the model's tokenizer.
Simulates rollout by generating real model responses via forward_logits + argmax.

Tests multiple batch sizes to find the single-GPU limit.
"""

import gc
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

sys.path.insert(0, "/media/volume/Model/MegaTrain")
sys.path.insert(0, "/media/volume/Model/verl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo_test")


# ── Configs ─────────────────────────────────────────────────────────────

MODEL_PATH = "/media/volume/Model/models--Qwen--Qwen3.5-27B/snapshots/b7ca741b86de18df552fd2cc952861e04621a4bd/"
GSM8K_PATH = "/media/volume/dataset/xma8/hf_cache/hub/datasets--openai--gsm8k/snapshots/cc7b047b6e5bb11b4f1af84efc572db110a51b3c/main"

@dataclass
class ModelConfig:
    path: str = MODEL_PATH
    trust_remote_code: bool = True

@dataclass
class EngineConfig:
    strategy: str = "megatrain"
    device: int = 0
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    checkpoint_interval: int = 4
    num_grad_slabs: int = 12
    max_seq_len: int = 2048
    forward_only: bool = False

@dataclass
class OptimizerConfig:
    lr: float = 1e-6
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    total_training_steps: int = 100
    lr_warmup_steps: int = 0
    lr_scheduler_type: str = "constant"

@dataclass
class CheckpointConfig:
    pass


def compute_grpo_advantage(token_level_rewards, response_mask, group_ids, epsilon=1e-6):
    """Compute GRPO advantage: normalize rewards within each group."""
    scores = token_level_rewards.sum(dim=-1)
    id2scores = defaultdict(list)
    for i in range(scores.shape[0]):
        id2scores[group_ids[i]].append(scores[i])
    id2mean, id2std = {}, {}
    for idx, s_list in id2scores.items():
        t = torch.stack(s_list)
        id2mean[idx] = t.mean()
        id2std[idx] = t.std() if len(s_list) > 1 else torch.tensor(1.0)
    with torch.no_grad():
        for i in range(scores.shape[0]):
            scores[i] = (scores[i] - id2mean[group_ids[i]]) / (id2std[group_ids[i]] + epsilon)
        advantages = scores.unsqueeze(-1) * response_mask
    return advantages


def prepare_real_data(tokenizer, num_prompts, n_per_prompt, max_prompt_len, max_response_len):
    """Load real GSM8K prompts and tokenize them. Pad responses with real-looking tokens."""
    import pandas as pd

    df = pd.read_parquet(os.path.join(GSM8K_PATH, "train-00000-of-00001.parquet"))

    instruction = 'Let\'s think step by step and output the final answer after "####".'
    batch_size = num_prompts * n_per_prompt

    all_input_ids = []
    all_attention_mask = []
    all_response_mask = []
    prompt_lens = []

    for p_idx in range(num_prompts):
        row = df.iloc[p_idx]
        question = row["question"] + " " + instruction

        # Apply chat template
        messages = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Truncate prompt if needed
        if len(prompt_tokens) > max_prompt_len:
            prompt_tokens = prompt_tokens[:max_prompt_len]
        actual_prompt_len = len(prompt_tokens)

        # Use the real answer as "response" tokens
        answer_text = row["answer"]
        answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
        if len(answer_tokens) > max_response_len:
            answer_tokens = answer_tokens[:max_response_len]

        # For each rollout copy, slightly vary the response
        for r_idx in range(n_per_prompt):
            resp_tokens = list(answer_tokens)
            # Add small variation: shift some tokens for different rollouts
            if r_idx > 0:
                # Simulate different model responses by shuffling/replacing some tokens
                n_vary = max(1, len(resp_tokens) // 10)
                for v in range(n_vary):
                    pos = (r_idx * 7 + v * 13) % len(resp_tokens)
                    resp_tokens[pos] = (resp_tokens[pos] + r_idx * 17) % tokenizer.vocab_size

            # Pad response to max_response_len
            resp_len = len(resp_tokens)
            if resp_len < max_response_len:
                resp_tokens = resp_tokens + [tokenizer.pad_token_id or 0] * (max_response_len - resp_len)
            else:
                resp_tokens = resp_tokens[:max_response_len]
                resp_len = max_response_len

            # Pad prompt to max_prompt_len
            if actual_prompt_len < max_prompt_len:
                padded_prompt = [tokenizer.pad_token_id or 0] * (max_prompt_len - actual_prompt_len) + prompt_tokens
            else:
                padded_prompt = prompt_tokens[:max_prompt_len]

            full_seq = padded_prompt + resp_tokens
            total_len = max_prompt_len + max_response_len

            attn = [0] * (max_prompt_len - actual_prompt_len) + [1] * (actual_prompt_len + resp_len) + [0] * (max_response_len - resp_len)
            resp_m = [0] * max_prompt_len + [1] * resp_len + [0] * (max_response_len - resp_len)

            all_input_ids.append(full_seq[:total_len])
            all_attention_mask.append(attn[:total_len])
            all_response_mask.append(resp_m[:total_len])
            prompt_lens.append(actual_prompt_len)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    response_mask = torch.tensor(all_response_mask, dtype=torch.float32)
    group_ids = np.repeat(np.arange(num_prompts), n_per_prompt)

    # Simulate rewards (correct answer = +1, wrong = -1, with noise)
    token_level_rewards = torch.zeros(batch_size, max_prompt_len + max_response_len)
    for i in range(batch_size):
        # First rollout of each group gets higher reward (simulates "correct")
        if i % n_per_prompt == 0:
            reward = 1.0 + torch.randn(1).item() * 0.1
        else:
            reward = -0.5 + torch.randn(1).item() * 0.5
        # Place reward at last valid response token
        valid_resp = response_mask[i].sum().int().item()
        if valid_resp > 0:
            last_resp_pos = max_prompt_len + valid_resp - 1
            token_level_rewards[i, last_resp_pos] = reward

    return input_ids, attention_mask, response_mask, group_ids, token_level_rewards, prompt_lens


def run_grpo_test(engine, tokenizer, num_prompts, n_per_prompt, max_prompt_len, max_response_len):
    """Run one full GRPO step and return timing + memory stats."""
    batch_size = num_prompts * n_per_prompt
    seq_len = max_prompt_len + max_response_len

    log.info(f"\n{'='*60}")
    log.info(f"  BS={batch_size} ({num_prompts}x{n_per_prompt}), seq={seq_len} "
             f"(prompt≤{max_prompt_len} + resp≤{max_response_len})")
    log.info(f"{'='*60}")

    # ── Prepare real data ──
    log.info("[1/5] Preparing real GSM8K data ...")
    input_ids, attention_mask, response_mask, group_ids, token_level_rewards, prompt_lens = \
        prepare_real_data(tokenizer, num_prompts, n_per_prompt, max_prompt_len, max_response_len)
    avg_prompt = np.mean(prompt_lens)
    avg_resp = response_mask.sum(dim=-1).mean().item()
    log.info(f"  avg prompt len: {avg_prompt:.0f}, avg response len: {avg_resp:.0f}")
    log.info(f"  total tokens per sample: {seq_len}, batch tokens: {batch_size * seq_len}")

    # ── Old log_probs ──
    log.info("[2/5] Computing old_log_probs (actor forward-only) ...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    from verl.utils.torch_functional import logprobs_from_logits

    logits = engine.cpu_master.forward_logits(input_ids, attention_mask)
    input_ids_gpu = input_ids.to(logits.device)
    log.info(f"  logits nan count: {logits.isnan().sum().item()}, inf count: {logits.isinf().sum().item()}")
    old_log_probs = logprobs_from_logits(logits[:, :-1, :], input_ids_gpu[:, 1:])
    old_log_probs = torch.cat([old_log_probs, torch.zeros(batch_size, 1, device=old_log_probs.device)], dim=1)
    log.info(f"  old_log_probs nan: {old_log_probs.isnan().sum().item()}, "
             f"inf: {old_log_probs.isinf().sum().item()}, "
             f"min: {old_log_probs[~old_log_probs.isnan()].min().item():.2f}, "
             f"max: {old_log_probs[~old_log_probs.isnan()].max().item():.2f}")

    del logits
    torch.cuda.synchronize()
    gpu_infer = torch.cuda.max_memory_allocated() / 1e9
    t_infer = time.time() - t0
    log.info(f"  Time: {t_infer:.2f}s, GPU peak: {gpu_infer:.2f} GB")

    # ── Ref log_probs ──
    log.info("[3/5] Computing ref_log_probs ...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    logits_ref = engine.cpu_master.forward_logits(input_ids, attention_mask)
    input_ids_gpu2 = input_ids.to(logits_ref.device)
    ref_log_probs = logprobs_from_logits(logits_ref[:, :-1, :], input_ids_gpu2[:, 1:])
    ref_log_probs = torch.cat([ref_log_probs, torch.zeros(batch_size, 1, device=ref_log_probs.device)], dim=1)

    del logits_ref, input_ids_gpu, input_ids_gpu2
    torch.cuda.synchronize()
    gpu_ref = torch.cuda.max_memory_allocated() / 1e9
    t_ref = time.time() - t0
    log.info(f"  Time: {t_ref:.2f}s, GPU peak: {gpu_ref:.2f} GB")

    # ── GRPO advantage ──
    advantages = compute_grpo_advantage(token_level_rewards, response_mask, group_ids)

    # ── Forward+backward ──
    log.info("[4/5] Running GRPO forward+backward ...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    engine.optimizer_zero_grad()

    device = torch.device(f"cuda:{EngineConfig().device}")
    old_log_probs_gpu = old_log_probs.detach().to(device)
    ref_log_probs_gpu = ref_log_probs.detach().to(device)
    advantages_gpu = advantages.to(device)
    response_mask_gpu = response_mask.to(device)

    del old_log_probs, ref_log_probs
    gc.collect()
    torch.cuda.empty_cache()

    def grpo_loss_fn(model_output, data, dp_group=None):
        log_prob = model_output["log_probs"]  # [B, T-1]
        resp_mask = response_mask_gpu[:, 1:]  # [B, T-1]
        adv = advantages_gpu[:, 1:]           # [B, T-1]
        old_lp = old_log_probs_gpu[:, 1:]     # [B, T-1]
        ref_lp = ref_log_probs_gpu[:, 1:]     # [B, T-1]

        # Mask out padding: only compute loss on valid response tokens
        # Clamp log_prob differences to avoid exp overflow on padding
        lp_diff = (log_prob - old_lp) * resp_mask
        lp_diff = torch.clamp(lp_diff, -5.0, 5.0)  # prevent exp explosion
        ratio = torch.exp(lp_diff)
        # Set ratio to 1.0 where mask is 0 (no contribution to loss)
        ratio = torch.where(resp_mask.bool(), ratio, torch.ones_like(ratio))

        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * clipped_ratio
        pg_loss = torch.max(pg_loss1, pg_loss2)

        num_valid = resp_mask.sum().clamp(min=1)
        pg_loss_mean = (pg_loss * resp_mask).sum() / num_valid

        # KL loss: only on valid response tokens
        ref_diff = (ref_lp - log_prob) * resp_mask
        ref_diff = torch.clamp(ref_diff, -5.0, 5.0)
        kl = torch.exp(ref_diff) - ref_diff - 1.0
        kl_loss = (kl * resp_mask).sum() / num_valid

        total_loss = pg_loss_mean + 0.001 * kl_loss

        with torch.no_grad():
            clip_frac = ((ratio[resp_mask.bool()] - 1.0).abs() > 0.2).float().mean() if resp_mask.any() else torch.tensor(0.0)
            approx_kl = (0.5 * lp_diff.pow(2) * resp_mask).sum() / num_valid

        return total_loss, {
            "pg_loss": pg_loss_mean.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "clip_frac": clip_frac.item(),
            "approx_kl": approx_kl.item(),
        }

    from tensordict import TensorDict
    batch_data = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, batch_size=batch_size)

    output = engine.forward_backward_batch(batch_data, grpo_loss_fn, forward_only=False)

    torch.cuda.synchronize()
    gpu_train = torch.cuda.max_memory_allocated() / 1e9
    t_train = time.time() - t0

    metrics = output.get("metrics", {})
    log.info(f"  pg_loss={metrics.get('pg_loss', 'N/A'):.4f}, "
             f"kl_loss={metrics.get('kl_loss', 'N/A'):.4f}, "
             f"total_loss={metrics.get('total_loss', 'N/A'):.4f}")
    log.info(f"  Time: {t_train:.2f}s, GPU peak: {gpu_train:.2f} GB")

    # ── Optimizer step ──
    log.info("[5/5] Optimizer step ...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    grad_norm = engine.optimizer_step()

    torch.cuda.synchronize()
    gpu_optim = torch.cuda.max_memory_allocated() / 1e9
    t_optim = time.time() - t0
    log.info(f"  grad_norm: {grad_norm:.4f}, Time: {t_optim:.2f}s")

    # Clean up GPU tensors
    del old_log_probs_gpu, ref_log_probs_gpu, advantages_gpu, response_mask_gpu, batch_data
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "bs": batch_size,
        "seq_len": seq_len,
        "gpu_infer": gpu_infer,
        "gpu_ref": gpu_ref,
        "gpu_train": gpu_train,
        "gpu_optim": gpu_optim,
        "t_infer": t_infer,
        "t_ref": t_ref,
        "t_train": t_train,
        "t_optim": t_optim,
        "t_total": t_infer + t_ref + t_train + t_optim,
        "grad_norm": grad_norm,
        "metrics": metrics,
        "status": "PASS" if (grad_norm > 0 and not np.isnan(grad_norm)) else "FAIL",
    }


def main():
    log.info("=" * 60)
    log.info("GRPO Training Test: Real GSM8K Data + Qwen3.5-27B")
    log.info("=" * 60)

    # ── Build engine ──
    log.info("\n[INIT] Building MegaTrain engine ...")
    t0 = time.time()

    from verl.workers.engine.megatrain.transformer_impl import MegaTrainEngine

    engine = MegaTrainEngine(
        model_config=ModelConfig(),
        engine_config=EngineConfig(),
        optimizer_config=OptimizerConfig(),
        checkpoint_config=CheckpointConfig(),
    )
    engine.initialize()
    torch.cuda.synchronize()
    t_init = time.time() - t0
    log.info(f"  Engine initialized in {t_init:.1f}s")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # ── Run tests with increasing batch sizes ──
    # max_prompt_len=256, max_response_len=768 → seq=1024
    test_configs = [
        # (num_prompts, n_per_prompt, max_prompt_len, max_response_len)
        (2,  4,  256, 768),   # BS=8,  seq=1024
        (4,  4,  256, 768),   # BS=16, seq=1024
        (8,  4,  256, 768),   # BS=32, seq=1024
        (16, 4,  256, 768),   # BS=64, seq=1024
    ]

    results = []
    for num_prompts, n_per, p_len, r_len in test_configs:
        try:
            r = run_grpo_test(engine, tokenizer, num_prompts, n_per, p_len, r_len)
            results.append(r)
        except torch.cuda.OutOfMemoryError as e:
            bs = num_prompts * n_per
            log.info(f"\n  *** OOM at BS={bs}, seq={p_len + r_len}: {e}")
            results.append({"bs": bs, "seq_len": p_len + r_len, "status": "OOM"})
            torch.cuda.empty_cache()
            gc.collect()
            break
        except Exception as e:
            bs = num_prompts * n_per
            log.info(f"\n  *** Error at BS={bs}: {e}")
            results.append({"bs": bs, "seq_len": p_len + r_len, "status": f"ERROR: {e}"})
            break

    # ── Summary table ──
    log.info("\n")
    log.info("=" * 80)
    log.info("BENCHMARK SUMMARY: Qwen3.5-27B GRPO (Real GSM8K Data)")
    log.info("=" * 80)
    log.info(f"{'BS':>4} {'SeqLen':>6} {'Status':>6} {'Infer':>7} {'Ref':>7} {'FwdBwd':>7} {'Optim':>7} {'Total':>7} {'GPU Peak':>9}")
    log.info("-" * 80)
    for r in results:
        if r["status"] == "PASS":
            gpu_peak = max(r["gpu_infer"], r["gpu_ref"], r["gpu_train"])
            log.info(f"{r['bs']:>4} {r['seq_len']:>6} {r['status']:>6} "
                     f"{r['t_infer']:>6.1f}s {r['t_ref']:>6.1f}s {r['t_train']:>6.1f}s "
                     f"{r['t_optim']:>6.1f}s {r['t_total']:>6.1f}s {gpu_peak:>8.2f}GB")
        else:
            log.info(f"{r['bs']:>4} {r['seq_len']:>6} {r['status']:>6}")
    log.info("=" * 80)

    engine.cleanup()
    log.info("Done.")


if __name__ == "__main__":
    main()
