# engine/trainer.py – Training loop with validation, grad clipping, AMP

import math
import torch
from contextlib import nullcontext


def configure_optimizer(model, config):
    """
    Create an AdamW optimizer with separate param groups:
      - Parameters that should be weight-decayed (Linear weights, >=2D)
      - Parameters that should NOT be decayed (biases, norms, embeddings, 1D)
    """
    # Separate parameters into decayed and non-decayed groups
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

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


@torch.no_grad()
def estimate_loss(model, val_loader, config):
    """Run a few batches of validation and return the mean loss."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= config.eval_iters:
            break
        x, y = x.to(config.device), y.to(config.device)
        _, loss = model(x, y)
        if loss is not None:
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def train(model, train_loader, val_loader, config):
    """Main training loop."""
    optimizer = configure_optimizer(model, config)

    # MPS / CUDA GradScaler setup
    use_cuda_scaler = config.use_amp and config.device == "cuda"
    if use_cuda_scaler:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        # Fallback dummy scaler for CPU / MPS where GradScaler is not needed
        class DummyScaler:
            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

        scaler = DummyScaler()

    model.to(config.device)
    model.train()
    best_val_loss = float("inf")

    # Infinite iterator over train_loader
    data_iter = iter(train_loader)

    print(f"Starting training on device: {config.device} for {config.max_iters} iterations...")

    for iter_num in range(config.max_iters):
        # a) Update learning rate
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # b) Fetch a batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(config.device), y.to(config.device)

        # c) Forward pass with AMP
        device_type = "cuda" if "cuda" in config.device else ("mps" if "mps" in config.device else "cpu")
        amp_ctx = (
            torch.amp.autocast(device_type=device_type, dtype=torch.float16)
            if config.use_amp and device_type in ("cuda", "mps")
            else nullcontext()
        )

        with amp_ctx:
            logits, loss = model(x, y)

        # d) Backward pass with scaling
        scaler.scale(loss).backward()

        # e) Gradient clipping
        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # f) Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # g) Logging
        if iter_num % 100 == 0:
            print(f"iter {iter_num:5d} | loss {loss.item():.4f} | lr {lr:.2e}")

        # h) Validation + checkpointing
        if iter_num % config.eval_interval == 0 and iter_num > 0:
            val_loss = estimate_loss(model, val_loader, config)
            print(f"  -> val loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")
                print(f"  -> saved best checkpoint (val_loss={val_loss:.4f})")

    # Final save
    torch.save(model.state_dict(), "final_model.pt")
    print("Training complete! Saved final_model.pt")
