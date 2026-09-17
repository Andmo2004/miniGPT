# engine/sampler.py – Text generation with temperature, top-k, top-p

import torch
import torch.nn.functional as F 
# TODO: Implement text generation strategies used at inference time.

#       Why is this separate from model.generate()? 
#   model.generate() handles the core autoregressive loop.
#   This module wraps it with user-friendly options:
#   loading a checkpoint, setting up the tokenizer, and providing
#   top-p (nucleus) sampling which is a bit more involved.

#   FUNCTION: top_k_filtering(logits, top_k) #
def top_k_filtering(logits, top_k):
    """Zero out all logits below the top-k threshold."""
    # TODO:
    
    #   1. Clamp top_k to at most vocab size:
    top_k = min(top_k, logits.size(-1))
    
    #   2. Find the k-th largest value in each row:
    values, _ = torch.topk(logits, top_k)
    threshold = values[:, -1].unsqueeze(-1)   # (B, 1)
    
    #   3. Mask everything below:
    logits[logits < threshold] = float('-inf')
    
    return logits

#  FUNCTION: top_p_filtering(logits, top_p) 
def top_p_filtering(logits, top_p):
    """
    Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability exceeds top_p.

    Example: if top_p = 0.9, keep tokens that together have 90% of the
    probability mass, discard the rest.
    """
    # TODO:
    
    #   1. Sort logits in descending order:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    #   2. Compute cumulative softmax probabilities:
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    #   3. Find tokens to remove (cumulative prob > top_p):
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    # (we subtract the current token's prob so that the first token
    # exceeding top_p is still included)
    
    #   4. Set removed tokens to -inf:
    sorted_logits[sorted_mask] = float('-inf')
    
    #   5. Unsort back to original order:
    logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
    return logits

#  FUNCTION: generate_text(model, tokenizer, prompt, ...) 
def generate_text(model, encode_fn, decode_fn, prompt, max_new_tokens=200,temperature=0.8, top_k=None, top_p=None, device='cpu'):
    """
    Full generation pipeline: encode prompt -> generate -> decode.
    """
    # TODO:
    
    #   1. Encode the prompt string into token IDs:
    token_ids = encode_fn(prompt)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    #   2. Set model to eval mode:
    model.eval()
    
    #   3. Use model.generate() or implement the loop here:
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
    
            if top_k is not None:
                logits = top_k_filtering(logits, top_k)
            if top_p is not None:
                logits = top_p_filtering(logits, top_p)
    
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
    
    #   4. Decode back to text:
    generated_ids = idx[0].tolist()
    
    return decode_fn(generated_ids)
