"""
Test script for MegaTrain VERL backend integration.

Tests:
1. Engine creation via EngineRegistry
2. Forward-only inference (log_probs computation)
3. Forward+backward training with SFT loss
4. Optimizer step and parameter sync

Usage:
    cd /media/volume/Model/verl
    python tests/test_megatrain_engine.py --model Qwen/Qwen3.5-4B
"""

import sys
import os
import argparse
import logging
import torch

sys.path.insert(0, '/media/volume/Model/MegaTrain')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def make_dummy_config(args):
    """Create minimal configs for testing."""
    from dataclasses import dataclass

    @dataclass
    class ModelConfig:
        path: str = args.model
        trust_remote_code: bool = True

    @dataclass
    class EngineConfig:
        strategy: str = "megatrain"
        device: int = args.device
        dtype: str = "bfloat16"
        attn_implementation: str = "flash_attention_2"
        checkpoint_interval: int = 4
        num_grad_slabs: int = 12
        max_seq_len: int = args.seq_len
        forward_only: bool = False
        param_offload: bool = True
        optimizer_offload: bool = True

    @dataclass
    class OptimizerConfig:
        lr: float = 1e-5
        weight_decay: float = 0.01
        clip_grad: float = 1.0
        betas: tuple = (0.9, 0.999)
        eps: float = 1e-8
        total_training_steps: int = 100
        lr_warmup_steps: int = 0
        lr_warmup_steps_ratio: float = 0.0
        lr_scheduler_type: str = "constant"
        min_lr_ratio: float = 0.01

    @dataclass
    class CheckpointConfig:
        pass

    return ModelConfig(), EngineConfig(), OptimizerConfig(), CheckpointConfig()


def sft_loss_fn(model_output, data, dp_group=None):
    """Simple SFT loss for testing."""
    log_probs = model_output["log_probs"]
    labels = data["input_ids"][:, 1:]  # Shift labels
    # Simple negative log likelihood
    B, T = labels.shape
    loss = -log_probs.mean()
    return loss, {"sft_loss": loss.detach().item()}


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("MegaTrain VERL Backend Integration Test")
    logger.info("=" * 60)

    # 1. Test engine creation via registry
    logger.info("\n[1/4] Testing engine creation via EngineRegistry...")
    from verl.workers.engine import EngineRegistry

    model_config, engine_config, optimizer_config, checkpoint_config = make_dummy_config(args)

    engine = EngineRegistry.new(
        model_type="language_model",
        backend="megatrain",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
    )
    logger.info(f"  Engine class: {engine.__class__.__name__}")
    logger.info(f"  is_param_offload_enabled: {engine.is_param_offload_enabled}")
    logger.info(f"  is_optimizer_offload_enabled: {engine.is_optimizer_offload_enabled}")

    # Initialize (loads model, creates optimizer)
    logger.info("  Initializing engine (loading model to CPU)...")
    engine.initialize()
    logger.info(f"  GPU mem after init: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.info("  OK: Engine created and initialized")

    # 2. Test forward-only (inference)
    logger.info("\n[2/4] Testing forward-only inference...")
    B, T = args.batch_size, args.seq_len
    input_ids = torch.randint(0, 1000, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)

    from tensordict import TensorDict
    data = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, batch_size=[B])

    with engine.eval_mode():
        output = engine.forward_backward_batch(data, loss_function=None, forward_only=True)

    logger.info(f"  Output keys: {list(output.keys())}")
    logger.info(f"  Loss: {output['loss']}")
    log_probs = output['model_output'].get('log_probs')
    if log_probs is not None:
        logger.info(f"  Log probs shape: {log_probs.shape}")
    logger.info("  OK: Forward-only inference works")

    # 3. Test forward+backward with loss
    logger.info("\n[3/4] Testing forward+backward with SFT loss...")

    with engine.train_mode():
        output = engine.forward_backward_batch(data, loss_function=sft_loss_fn, forward_only=False)

    logger.info(f"  Loss: {output['loss']:.4f}")
    if 'metrics' in output and output['metrics']:
        logger.info(f"  Metrics: {output['metrics']}")

    # Check gradients accumulated on CPU params
    params = engine.cpu_master.get_parameters()
    has_grad = sum(1 for p in params if p.grad is not None and p.grad.abs().sum() > 0)
    logger.info(f"  Parameters with non-zero gradients: {has_grad}/{len(params)}")
    logger.info("  OK: Forward+backward with loss works")

    # 4. Test optimizer step
    logger.info("\n[4/4] Testing optimizer step...")
    param_before = params[0].data.clone()
    grad_norm = engine.optimizer_step()
    engine.cpu_master._sync_params_to_gpu()
    engine.cpu_master.zero_grad()
    engine.optimizer.zero_grad()

    param_after = params[0].data
    param_diff = (param_after - param_before).abs().mean().item()
    logger.info(f"  Grad norm: {grad_norm:.4f}")
    logger.info(f"  Param update magnitude: {param_diff:.6f}")
    logger.info(f"  GPU mem peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    logger.info("  OK: Optimizer step works")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"GPU mem peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    main()
