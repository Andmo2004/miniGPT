# engine/trainer.py – Training loop with validation, grad clipping, AMP
#                                                        ★ HARD FILE ★
#
# TODO: Implement the training engine that orchestrates the full training
#       process: optimizer setup, learning rate scheduling, training step,
#       validation loop, checkpointing, and logging.
#
# ══════════════════════════════════════════════════════════════════════════
# ── FUNCTION: configure_optimizer(model, config) ──
# ══════════════════════════════════════════════════════════════════════════
#
#   def configure_optimizer(model, config):
#       """
#       Create an AdamW optimizer with separate param groups:
#         - Parameters that should be weight-decayed (Linear weights, ≥2D)
#         - Parameters that should NOT be decayed (biases, norms, embeddings, 1D)
#       """
#       # TODO:
#       #   1. Iterate over model.named_parameters().
#       #   2. Separate into two groups:
#       #        decay_params   = [p for n, p in ... if p.dim() >= 2]
#       #        no_decay_params = [p for n, p in ... if p.dim() < 2]
#       #   3. Create optimizer:
#       #        optim_groups = [
#       #            {"params": decay_params,    "weight_decay": config.weight_decay},
#       #            {"params": no_decay_params, "weight_decay": 0.0},
#       #        ]
#       #        optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate,
#       #                                      betas=(0.9, 0.95), eps=1e-8)
#       #   4. Print the number of decayed vs non-decayed params.
#       #   5. return optimizer
#       #
#       #   WHY separate groups?
#       #   Weight decay acts as L2 regularisation on weight matrices, but
#       #   applying it to bias terms and norm params can hurt training.
#
# ══════════════════════════════════════════════════════════════════════════
# ── FUNCTION: get_lr(iter_num, config) ──
# ══════════════════════════════════════════════════════════════════════════
#
#   def get_lr(iter_num, config):
#       """
#       Cosine learning rate schedule with linear warmup.
#
#       LR schedule looks like:
#         ╱‾‾‾╲
#        /      ╲___________
#       / warmup  cosine decay  → min_lr
#       """
#       # TODO:
#       #   min_lr = config.learning_rate / 10   # floor LR (10% of peak)
#       #
#       #   1. Linear warmup phase (iter_num < warmup_iters):
#       #        return config.learning_rate * (iter_num / config.warmup_iters)
#       #
#       #   2. After max_iters, return min_lr:
#       #        if iter_num > config.max_iters:
#       #            return min_lr
#       #
#       #   3. Cosine decay between warmup_iters and max_iters:
#       #        progress = (iter_num - config.warmup_iters) / (config.max_iters - config.warmup_iters)
#       #        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))   # goes from 1→0
#       #        return min_lr + coeff * (config.learning_rate - min_lr)
#
# ══════════════════════════════════════════════════════════════════════════
# ── FUNCTION: estimate_loss(model, val_loader, config) ──
# ══════════════════════════════════════════════════════════════════════════
#
#   @torch.no_grad()
#   def estimate_loss(model, val_loader, config):
#       """Run a few batches of validation and return the mean loss."""
#       # TODO:
#       #   1. Set model to eval mode: model.eval()
#       #   2. Accumulate losses over config.eval_iters batches:
#       #        losses = []
#       #        for i, (x, y) in enumerate(val_loader):
#       #            if i >= config.eval_iters:
#       #                break
#       #            x, y = x.to(config.device), y.to(config.device)
#       #            _, loss = model(x, y)
#       #            losses.append(loss.item())
#       #   3. Set model back to train mode: model.train()
#       #   4. return sum(losses) / len(losses)
#
# ══════════════════════════════════════════════════════════════════════════
# ── FUNCTION: train(model, train_loader, val_loader, config) ──
# ══════════════════════════════════════════════════════════════════════════
#
#   def train(model, train_loader, val_loader, config):
#       """Main training loop."""
#       # TODO:
#       #   1. Set up:
#       #        optimizer = configure_optimizer(model, config)
#       #
#       #        ⚠️ MPS NOTE: GradScaler is NOT supported on MPS.
#       #        Only enable it when running on CUDA:
#       #        use_scaler = config.use_amp and config.device == 'cuda'
#       #        scaler = torch.amp.GradScaler(enabled=use_scaler)
#       #
#       #        model.to(config.device)
#       #        model.train()
#       #        best_val_loss = float('inf')
#       #
#       #   2. Create an infinite iterator over the train_loader:
#       #        (since we train for a fixed number of *iterations*, not epochs)
#       #        data_iter = iter(train_loader)
#       #
#       #   3. Training loop:
#       #        for iter_num in range(config.max_iters):
#       #
#       #            a) Update learning rate:
#       #                 lr = get_lr(iter_num, config)
#       #                 for param_group in optimizer.param_groups:
#       #                     param_group['lr'] = lr
#       #
#       #            b) Fetch a batch (handle iterator exhaustion):
#       #                 try:
#       #                     x, y = next(data_iter)
#       #                 except StopIteration:
#       #                     data_iter = iter(train_loader)
#       #                     x, y = next(data_iter)
#       #                 x, y = x.to(config.device), y.to(config.device)
#       #
#       #            c) Forward pass with AMP:
#       #                 ⚠️ MPS NOTE: autocast works on MPS but you must pass
#       #                 the correct device_type string. For MPS use 'mps',
#       #                 for CUDA use 'cuda'. config.device already has the
#       #                 right value.
#       #
#       #                 with torch.amp.autocast(device_type=config.device,
#       #                                         dtype=torch.float16,
#       #                                         enabled=config.use_amp):
#       #                     logits, loss = model(x, y)
#       #
#       #            d) Backward pass with gradient scaling:
#       #                 ⚠️ MPS NOTE: If use_scaler is False (MPS), scaler.scale()
#       #                 is a no-op and just returns the loss unchanged, so this
#       #                 code works on both CUDA and MPS without branching.
#       #                 scaler.scale(loss).backward()
#       #
#       #            e) Gradient clipping (BEFORE optimizer step):
#       #                 if config.grad_clip > 0:
#       #                     scaler.unscale_(optimizer)
#       #                     torch.nn.utils.clip_grad_norm_(model.parameters(),
#       #                                                     config.grad_clip)
#       #
#       #            f) Optimizer step + scaler update:
#       #                 scaler.step(optimizer)
#       #                 scaler.update()
#       #                 optimizer.zero_grad(set_to_none=True)
#       #                    set_to_none=True is slightly more efficient than
#       #                    zeroing gradients (avoids a memset).
#       #
#       #            g) Logging:
#       #                 if iter_num % 100 == 0:
#       #                     print(f"iter {iter_num} | loss {loss.item():.4f} | lr {lr:.2e}")
#       #
#       #            h) Validation + checkpointing:
#       #                 if iter_num % config.eval_interval == 0 and iter_num > 0:
#       #                     val_loss = estimate_loss(model, val_loader, config)
#       #                     print(f"  → val loss: {val_loss:.4f}")
#       #                     if val_loss < best_val_loss:
#       #                         best_val_loss = val_loss
#       #                         torch.save(model.state_dict(), "best_model.pt")
#       #                         print(f"  → saved checkpoint (val_loss={val_loss:.4f})")
#       #
#       #   4. Final save:
#       #        torch.save(model.state_dict(), "final_model.pt")
#       #        print("Training complete!")
