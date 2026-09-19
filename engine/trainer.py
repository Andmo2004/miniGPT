# engine/trainer.py – Training loop with validation, grad clipping, AMP

import math
import os
import time
import torch
from contextlib import nullcontext


def configure_optimizer(model, config):
    """
    Create an AdamW optimizer with separate param groups:
      - Parameters that should be weight-decayed (Linear weights, >=2D)
      - Parameters that should NOT be decayed (biases, norms, embeddings, 1D)
    """
    # Separate parameters into decayed and non-decayed groups.
    # Weight decay should apply to Linear weights (>=2D) but NOT to:
    #   - biases (1D)
    #   - LayerNorm / RMSNorm parameters (1D)
    #   - Embedding weights (2D, but should not be decayed)
    # We check parameter names to correctly exclude embeddings.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Embeddings: 2D but should not be decayed
        if "emb" in name or isinstance(
            dict(model.named_modules()).get(name.rsplit(".", 1)[0]),
            torch.nn.Embedding,
        ):
            no_decay_params.append(param)
        elif param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in no_decay_params)
    print(f"Decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"Non-decayed parameter tensors: {len(no_decay_params)}, with {num_nodecay_params:,} parameters")

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return optimizer


def get_lr(iter_num, config):
    """
    Cosine learning rate schedule with linear warmup.
    """
    min_lr = config.learning_rate / 10.0  # floor LR (10% of peak)

    # 1. Linear warmup phase
    if iter_num < config.warmup_iters:
        return config.learning_rate * (iter_num / max(1, config.warmup_iters))

    # 2. After max_iters, return min_lr
    if iter_num > config.max_iters:
        return min_lr

    # 3. Cosine decay between warmup_iters and max_iters
    progress = (iter_num - config.warmup_iters) / max(1, (config.max_iters - config.warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (config.learning_rate - min_lr)


def get_amp_settings(config):
    """
    Determine AMP dtype and whether to use GradScaler based on device
    and hardware capabilities.
    
    Returns:
        (device_type, amp_dtype, use_grad_scaler)
    """
    device_type = "cuda" if "cuda" in config.device else (
        "mps" if "mps" in config.device else "cpu"
    )

    if not config.use_amp:
        return device_type, torch.float32, False

    if device_type == "cuda":
        # Prefer BF16 on capable hardware (no GradScaler needed)
        if torch.cuda.is_bf16_supported():
            return device_type, torch.bfloat16, False
        else:
            return device_type, torch.float16, True
    elif device_type == "mps":
        # MPS supports autocast with float16, but NOT GradScaler
        return device_type, torch.float16, False
    else:
        return device_type, torch.float32, False


def save_checkpoint(model, optimizer, scaler, iter_num, best_val_loss, config,
                    filepath, use_grad_scaler=False):
    """Save a full training checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config.to_dict(),
        "scaler": scaler.state_dict() if use_grad_scaler else None,
    }
    torch.save(checkpoint, filepath)
    print(f"  -> saved checkpoint to {filepath} (iter={iter_num}, val_loss={best_val_loss:.4f})")


def load_checkpoint(filepath, model, optimizer=None, scaler=None, device="cpu"):
    """
    Load a training checkpoint. Returns (iter_num, best_val_loss, config_dict).
    
    Model and optimizer states are loaded in-place. If optimizer or scaler
    are None, their states are skipped.
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    config_dict = checkpoint.get("config", {})

    print(f"Resumed from {filepath} at iter {iter_num} (best_val_loss={best_val_loss:.4f})")
    return iter_num, best_val_loss, config_dict


@torch.no_grad()
def estimate_loss(model, val_loader, config, amp_ctx):
    """Run a few batches of validation and return the mean loss."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= config.eval_iters:
            break
        x, y = x.to(config.device), y.to(config.device)
        with amp_ctx:
            _, loss = model(x, y)
        if loss is not None:
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def train(model, train_loader, val_loader, config, resume_from=None):
    """Main training loop with full checkpoint support, gradient accumulation,
    AMP auto-detection, and periodic saves.

    Checkpoints are saved to config.out_dir."""

    optimizer = configure_optimizer(model, config)

    # AMP setup with BF16/FP16 auto-detection
    device_type, amp_dtype, use_grad_scaler = get_amp_settings(config)

    if use_grad_scaler:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        # Dummy scaler for CPU / MPS / BF16 where GradScaler is not needed
        class DummyScaler:
            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                pass

        scaler = DummyScaler()

    # AMP autocast context
    if config.use_amp and device_type in ("cuda", "mps"):
        amp_ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype)
        print(f"AMP enabled: dtype={amp_dtype}, GradScaler={'ON' if use_grad_scaler else 'OFF'}")
    else:
        amp_ctx = nullcontext()

    model.to(config.device)
    model.train()
    best_val_loss = float("inf")
    start_iter = 0

    # Resume from checkpoint if requested
    if resume_from is not None and os.path.exists(resume_from):
        start_iter, best_val_loss, _ = load_checkpoint(
            resume_from, model, optimizer,
            scaler if use_grad_scaler else None,
            device=config.device,
        )
        start_iter += 1  # resume from the next iteration

    # Infinite iterator over train_loader
    data_iter = iter(train_loader)

    # Gradient accumulation setup
    grad_accum_steps = config.grad_accumulation_steps
    effective_batch_size = config.batch_size * grad_accum_steps

    print(f"Starting training on device: {config.device} for {config.max_iters} iterations...")
    print(f"Effective batch size: {effective_batch_size} "
          f"({config.batch_size} × {grad_accum_steps} accum steps)")

    # Create output directory for checkpoints
    out_dir = config.out_dir
    os.makedirs(out_dir, exist_ok=True)

    tokens_seen = 0
    t0 = time.time()

    for iter_num in range(start_iter, config.max_iters):
        # a) Update learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # b) Gradient accumulation loop
        accumulated_loss = 0.0
        for micro_step in range(grad_accum_steps):
            # Fetch a batch
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(config.device), y.to(config.device)
            tokens_seen += x.numel()

            # Forward pass with AMP
            with amp_ctx:
                logits, loss = model(x, y)
                # Scale loss by accumulation steps so gradients average correctly
                loss = loss / grad_accum_steps

            # Backward pass with scaling
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        # c) Gradient clipping
        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # d) Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # e) Logging
        if iter_num % config.log_interval == 0:
            dt = time.time() - t0
            tokens_per_sec = tokens_seen / dt if dt > 0 else 0

            log_msg = (f"iter {iter_num:5d} | loss {accumulated_loss:.4f} "
                       f"| lr {lr:.2e} | {tokens_per_sec:.0f} tok/s")

            # GPU memory logging
            if "cuda" in config.device:
                mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                log_msg += f" | GPU mem {mem_mb:.0f}MB"

            print(log_msg)

        # f) Validation + best checkpoint
        if iter_num % config.eval_interval == 0 and iter_num > 0:
            val_loss = estimate_loss(model, val_loader, config, amp_ctx)
            print(f"  -> val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scaler, iter_num, best_val_loss, config,
                    os.path.join(out_dir, "best_model.pt"), use_grad_scaler,
                )

        # g) Periodic checkpoint
        if iter_num % config.save_interval == 0 and iter_num > 0:
            save_checkpoint(
                model, optimizer, scaler, iter_num, best_val_loss, config,
                os.path.join(out_dir, f"checkpoint_{iter_num}.pt"), use_grad_scaler,
            )

    # Final evaluation if no validation was performed yet or if eval_interval divides max_iters
    if best_val_loss == float("inf") or config.max_iters % config.eval_interval == 0:
        val_loss = estimate_loss(model, val_loader, config, amp_ctx)
        print(f"  -> final val loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scaler, config.max_iters, best_val_loss, config,
                os.path.join(out_dir, "best_model.pt"), use_grad_scaler,
            )

    # Final save
    final_path = os.path.join(out_dir, "final_model.pt")
    best_path = os.path.join(out_dir, "best_model.pt")
    save_checkpoint(
        model, optimizer, scaler, config.max_iters, best_val_loss, config,
        final_path, use_grad_scaler,
    )
    if not os.path.exists(best_path):
        save_checkpoint(
            model, optimizer, scaler, config.max_iters, best_val_loss, config,
            best_path, use_grad_scaler,
        )

    total_time = time.time() - t0
    print(f"Training complete! Total time: {total_time:.1f}s")
    print(f"Total tokens processed: {tokens_seen:,}")
