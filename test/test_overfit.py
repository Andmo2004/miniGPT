# test/test_overfit.py – Single-batch overfit sanity test
#
# TODO: Write a test that verifies the model can overfit a single batch.
#
#   This is the most important sanity check in deep learning:
#   If your model can't memorise ONE batch, something is fundamentally
#   broken (wrong loss, wrong shapes, bad init, etc.).
#
#   def test_overfit_single_batch():
#       # TODO:
#       #   1. Create a tiny config:
#       #        cfg = GPTConfig(vocab_size=100, d_model=64, n_heads=4,
#       #                        n_layers=2, block_size=32, dropout=0.0)
#       #        ⚠️ dropout=0.0 is critical here — dropout is stochastic
#       #           and will prevent perfect memorisation.
#       #
#       #   2. Create model + optimizer:
#       #        model = GPT(cfg)
#       #        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#       #
#       #   3. Create ONE random batch:
#       #        x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
#       #        y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
#       #
#       #   4. Train on that single batch for ~200 steps:
#       #        for step in range(200):
#       #            logits, loss = model(x, y)
#       #            optimizer.zero_grad()
#       #            loss.backward()
#       #            optimizer.step()
#       #
#       #   5. Assert loss is very small:
#       #        assert loss.item() < 0.1, f"Loss didn't converge: {loss.item()}"
#       #
#       #   If this fails, debug:
#       #     - Is the loss going down at all? (print every 50 steps)
#       #     - Is the cross_entropy applied correctly?
#       #     - Are the shapes of logits/targets correct for F.cross_entropy?
#
# ── Run with: pytest test/test_overfit.py -v -s
