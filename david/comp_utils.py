import torch
from tqdm import tqdm

def get_activations_slow(prompts, model, tokenizer):
    activations = []
    for prompt in tqdm(prompts):
        tokens = model.to_tokens(prompt)
        logits, cache = model.run_with_cache(prompt, 
                                             names_filter = lambda name : name.endswith("resid_post"))
        act = torch.stack(list(cache.values()))[:, 0, -1] # (nlayer, hidden_size)
        activations.append(act)
    return torch.stack(activations, dim=0) # (nprompt, nlayer, hidden_size)

# %%
@torch.no_grad()
def get_activations(prompts, model, tokenizer, batch_size=32):
    # Create a list of (index, prompt) tuples
    indexed_prompts = list(enumerate(prompts))
    
    # Sort by prompt length, but keep original indices
    sorted_indexed_prompts = sorted(indexed_prompts, key=lambda x: len(x[1]), reverse=True)
    
    activations = []
    total_prompts = len(prompts)
    
    with tqdm(total=total_prompts, desc="Processing prompts") as pbar:
        for i in range(0, total_prompts, batch_size):
            batch = sorted_indexed_prompts[i:i+batch_size]
            
            # Separate indices and prompts
            indices, batch_prompts = zip(*batch)
            
            # Use model.to_tokens for tokenization (automatically adds BOS and pads)
            tokens = model.to_tokens(batch_prompts)
            
            # Find the last non-padding token position
            bos_token_id = tokenizer.bos_token_id
            last_token_pos = torch.zeros(tokens.shape[0], dtype=torch.long)
            for i, seq in enumerate(tokens):
                # Find all occurrences of BOS token
                bos_positions = (seq == bos_token_id).nonzero().flatten()
                # The last non-padding token is just before the second BOS token (if it exists)
                if len(bos_positions) > 1:
                    last_token_pos[i] = bos_positions[1] - 1
                else:
                    # If there's no second BOS token, the last token is the last non-BOS token
                    last_token_pos[i] = (seq != bos_token_id).nonzero().flatten()[-1]
            
            # Run the model with cache
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name.endswith("resid_post")
                )
            
            # Extract activations for the last token of each sequence
            batch_activations = []
            for layer_name, layer_cache in cache.items():
                # layer_cache shape: (batch_size, seq_len, hidden_size)
                last_token_activations = layer_cache[torch.arange(layer_cache.size(0)), last_token_pos]
                batch_activations.append(last_token_activations)
            
            # Stack activations for all layers
            batch_activations = torch.stack(batch_activations)  # (n_layers, batch_size, hidden_size)
            
            # Add to activations list along with original indices
            activations.extend(list(zip(indices, batch_activations.transpose(0, 1))))
            
            pbar.update(len(batch))
    
    # Sort activations back to original order
    sorted_activations = [act for _, act in sorted(activations, key=lambda x: x[0])]
    
    return torch.stack(sorted_activations)  # (n_prompts, n_layers, hidden_size)

# %%
def test_get_activations(test_prompts, model, tokenizer):
    cache_slow = get_activations_slow(test_prompts, model, tokenizer)
    cache_fast = get_activations(test_prompts, model, tokenizer)
    print(f"{(cache_slow - cache_fast).abs().max()=}")