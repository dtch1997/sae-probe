# %%
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
# %%
if locals().get("LOAD_MODEL") is None:
    model_name = "gpt2"
    model = HookedTransformer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    LOAD_MODEL = True
    # Load the sentiment dataset
    dataset = load_dataset("emotion", split="train")

# %%
#no gradients
torch.set_grad_enabled(False)


# Filter for happy and angry prompts
joy_prompts = dataset.filter(lambda example: example['label'] == 1)['text']  # 3 is the label for 'joy'
sad_prompts = dataset.filter(lambda example: example['label'] == 0)['text']  # 0 is the label for 'anger'
print(f"Number of happy prompts: {len(happy_prompts)}")
print(f"Number of angry prompts: {len(angry_prompts)}")

# %%

def get_activations(prompts, model, tokenizer):
    activations = []
    for prompt in tqdm(prompts):
        logits, cache = model.run_with_cache(prompt, 
                                             names_filter = lambda name : name.endswith("resid_post"))
        act = torch.stack(list(cache.values()))[:, 0, -1] # (nlayer, hidden_size)
        activations.append(act)
    return torch.stack(activations, dim=0) # (nprompt, nlayer, hidden_size)


# Get prompts for X and Y
print("Happy prompts:\n")
print("\n".join(joy_prompts[:5]))
print("\nSad prompts:\n")
print("\n".join(sad_prompts[:5]))

# %%
joy_prompts_activations = get_activations(joy_prompts, model, tokenizer)
# %%
