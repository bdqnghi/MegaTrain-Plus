"""End-to-end training smoke test: 3 steps with optimizer.step() actually running.

Confirms that MegaTrain-Plus defaults (store_all_activations, zero_copy_unflatten,
backward_prefetch, num_buffers=3) work through a full training loop, not just
the forward+backward path the benchmark harness isolates.

Verifies:
  1. Loss decreases across steps (training actually works).
  2. Gradients flow (grad_norm is finite).
  3. Optimizer.step() completes without crashes or warnings.
  4. Parameters update on CPU and the GPU templates see the new values
     on the next step (no stale-weight bug from zero-copy aliasing).

Uses Qwen2.5-0.5B for speed.
"""

import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

from infinity import CPUMasterModel, ChatDataset, collate_fn
from infinity.config import CPUMasterConfig


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    log.info(f"Smoke test: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to("cpu")

    config = CPUMasterConfig(
        model_name=model_name, batch_size=4, max_seq_len=512, num_steps=3,
        dataset_name="alpaca_en_demo", dataset_dir="data",
        # MegaTrain-Plus defaults
        num_buffers=3,
        backward_prefetch=True,
        store_all_activations=True,
        zero_copy_unflatten=True,
        num_grad_workers=2,
    )
    model = CPUMasterModel(hf_model, config)
    del hf_model

    log.info("Creating optimizer (PyTorch AdamW on CPU params)...")
    optimizer = torch.optim.AdamW(model.get_parameters(), lr=1e-4, weight_decay=0.01)

    dataset = ChatDataset(tokenizer, config.max_seq_len, dataset_name="alpaca_en_demo", dataset_dir="data")
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, collate_fn=collate_fn, num_workers=0
    )
    # Overfit test: grab one batch and train on it repeatedly.
    # Loss on that fixed batch MUST decrease if optimizer.step() and
    # parameter updates are wired correctly. Training-on-different-batches
    # (different content) would not be a correctness test.
    fixed_batch = next(iter(dataloader))

    # Snapshot a few weights BEFORE training so we can verify they actually change
    probe_params = list(model.get_parameters())[:3]
    snapshots_before = [p.data.clone() for p in probe_params]

    log.info("Running 5 training steps on the SAME batch (overfit test)...")
    losses = []
    grad_norms = []
    for step in range(5):
        batch = fixed_batch

        t0 = time.perf_counter()
        loss_val, n_tokens, timing = model.forward_and_backward(
            batch["input_ids"], batch["attention_mask"], batch["labels"]
        )
        t_fb = time.perf_counter() - t0

        grad_norm = torch.nn.utils.clip_grad_norm_(model.get_parameters(), 1.0)
        t0 = time.perf_counter()
        optimizer.step()
        model._sync_params_to_gpu()
        model.zero_grad()
        optimizer.zero_grad()
        t_opt = time.perf_counter() - t0

        losses.append(loss_val)
        grad_norms.append(grad_norm.item() if torch.isfinite(grad_norm) else float("nan"))
        log.info(
            f"step {step}: loss={loss_val:.4f}  grad_norm={grad_norm.item():.3f}  "
            f"fwd+bwd={t_fb:.2f}s  opt={t_opt:.2f}s"
        )

    # --- Assertions ---
    print("\n=== Verification ===")

    # 1. Loss decreased on fixed-batch overfit test
    assert all(loss > 0 for loss in losses), f"Non-positive loss encountered: {losses}"
    log.info(f"Losses: {[f'{l:.4f}' for l in losses]}")
    # With lr=1e-4 on a fixed batch, loss MUST decrease materially over 5 steps.
    # If optimizer.step() were a no-op or parameters weren't syncing back to GPU,
    # loss would stay flat at the initial value.
    drop = losses[0] - losses[-1]
    assert drop > 0.05, (
        f"Loss did not decrease meaningfully on fixed-batch overfit. "
        f"step 0: {losses[0]:.4f}  step {len(losses)-1}: {losses[-1]:.4f}  drop={drop:.4f}"
    )
    print(f"  [PASS] Loss decreased on fixed batch: {losses[0]:.4f} -> {losses[-1]:.4f}  (drop {drop:.4f})")

    # 2. Grad norms were finite and nonzero
    assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms), f"Non-finite grad_norm: {grad_norms}"
    assert all(g > 0 for g in grad_norms), f"Zero grad_norm: {grad_norms}"
    print(f"  [PASS] grad_norms finite and positive: {[f'{g:.3f}' for g in grad_norms]}")

    # 3. Weights actually changed on CPU
    for i, (p, before) in enumerate(zip(probe_params, snapshots_before)):
        diff = (p.data - before).abs().max().item()
        assert diff > 0, f"Param {i} did not change after optimizer.step()!"
        print(f"  [PASS] param {i} changed: max abs diff = {diff:.6f}")

    # 4. GPU templates see the updated weights (ensures _sync_params_to_gpu works)
    # With zero-copy unflatten, GPU templates alias the flat buffer. After
    # _sync_params_to_gpu, the NEXT _load_layer_to_buffer_async fills the flat
    # buffer from the updated CPU params. Verify by loading and checking.
    if len(model.cpu_layers) > 0:
        model._load_layer_to_buffer_async(0, 0)
        model.weight_stream.synchronize()
        model._unflatten_to_layer(0, 0)
        torch.cuda.synchronize()
        gpu_layer = model._get_gpu_layer(0, 0)
        gpu_first_param = next(gpu_layer.parameters())
        cpu_first_param = next(model.cpu_layers[0].parameters())
        # They should be approximately equal (BF16 quantization noise OK)
        diff = (gpu_first_param.data - cpu_first_param.data.to("cuda")).abs().max().item()
        assert diff < 0.01, f"GPU template param diverged from CPU master by {diff}"
        print(f"  [PASS] GPU template matches updated CPU master (max abs diff={diff:.6f})")

    print("\n*** Training smoke test PASSED ***")
    model.cleanup()


if __name__ == "__main__":
    main()
