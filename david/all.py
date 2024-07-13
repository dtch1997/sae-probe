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
    print(f"{(cache_slow - cache_fast).abs().max()=}")# %%
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
    dataset = load_dataset("emotion", "unsplit")['train']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
#no gradients
torch.set_grad_enabled(True)
model = torch.compile(model)
# %%

# Filter for happy and angry prompts
raw_joy_prompts = dataset.filter(lambda example: example['label'] == 1)['text']  # 3 is the label for 'joy'
raw_sad_prompts = dataset.filter(lambda example: example['label'] == 0)['text']  # 0 is the label for 'anger'
print(f"Number of joy prompts: {len(raw_joy_prompts)}")
print(f"Number of sad prompts: {len(raw_sad_prompts)}")
dataset_len = min(len(raw_joy_prompts), len(raw_sad_prompts))
joy_prompts = raw_joy_prompts[:dataset_len]
sad_prompts = raw_sad_prompts[:dataset_len]
print(f"Number of prompts after truncation: {len(joy_prompts)}")

# %%

joy_act_path = './joy_act.pt'
sad_act_path = './sad_act.pt'

from utils import cache_function_call
from comp_utils import get_activations

torch.cuda.empty_cache()
joy_act = cache_function_call(
    joy_act_path, 
    get_activations, 
    use_torch=True, 
    prompts=joy_prompts, 
    model=model, 
    tokenizer=tokenizer, 
    batch_size=512
)

torch.cuda.empty_cache()
sad_act = cache_function_call(
    sad_act_path, 
    get_activations, 
    use_torch=True, 
    prompts=sad_prompts, 
    model=model, 
    tokenizer=tokenizer, 
    batch_size=512
)
# %%
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class LogisticRegressionProbe(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionProbe, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def train_probe(X, y, max_iter=100, tol=1e-4, batch_size=57344):
    model = LogisticRegressionProbe(X.shape[1]).cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    n_samples = X.shape[0]
    n_batches = (n_samples - 1) // batch_size + 1

    for epoch in range(max_iter):
        total_loss = 0
        optimizer.zero_grad()

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]

            with autocast():
                output = model(X_batch)
                loss = criterion(output, y_batch.unsqueeze(1))
                loss = loss / n_batches  # Normalize the loss

            loss.backward()
            total_loss += loss.item() * n_batches

        optimizer.step()
        
        # if total_loss < tol:
        #     break

    return model

def evaluate_probe(model, X, y):
    model.eval()
    with torch.no_grad(), autocast():
        logits = model(X)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, y.unsqueeze(1))
        y_pred = torch.sigmoid(logits)
        accuracy = ((y_pred > 0.5).squeeze() == y).float().mean()
    return bce_loss.item(), accuracy.item()

def analyze_all_layers(X, Y, test_size=0.2, batch_size=57344):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    num_layers = X.shape[1]
    num_features = X.shape[2]

    # Combine X and Y, and create labels
    combined_data = torch.cat([X, Y], dim=0)
    labels = torch.cat([torch.ones(X.shape[0]), torch.zeros(Y.shape[0])])

    # Perform train-test split
    train_data, test_data, train_labels, test_labels = train_test_split(
        combined_data.cpu().numpy(), labels.cpu().numpy(), 
        test_size=test_size, stratify=labels.cpu().numpy(), random_state=42
    )

    # Move data to GPU after split
    train_data = torch.from_numpy(train_data).float().to(device)
    test_data = torch.from_numpy(test_data).float().to(device)
    train_labels = torch.from_numpy(train_labels).float().to(device)
    test_labels = torch.from_numpy(test_labels).float().to(device)

    # Generate sample sizes
    max_samples = len(train_data)
    sample_sizes = torch.unique(torch.logspace(3, torch.log10(torch.tensor(max_samples)), steps=20).int())

    results = []

    for size in tqdm(sample_sizes, desc="Analyzing sample sizes"):
        # Create a subset of the training data
        subset_indices = torch.randperm(len(train_data))[:size]
        X_train_subset = train_data[subset_indices]
        y_train_subset = train_labels[subset_indices]

        # Analyze each layer
        layer_results = []
        for layer_idx in range(num_layers):
            X_train_layer = X_train_subset[:, layer_idx, :]
            
            # Ensure the training data requires gradients
            X_train_layer = X_train_layer.requires_grad_()
            
            # Train the probe
            model = train_probe(X_train_layer, y_train_subset, batch_size=batch_size)

            # Evaluate on train and test sets
            train_loss, train_acc = evaluate_probe(model, X_train_layer, y_train_subset)
            
            X_test_layer = test_data[:, layer_idx, :]
            test_loss, test_acc = evaluate_probe(model, X_test_layer, test_labels)

            layer_results.append((layer_idx, size.item(), train_acc, test_acc, train_loss, test_loss))

        results.extend(layer_results)

    return results



# %%


# Usage
# Assuming X and Y are your input tensors
# %%
if LOAD_MODEL:
    del model
    torch.cuda.empty_cache() #like 400 mb to spare lmao
    
results = cache_function_call('results.pkl', analyze_all_layers, X=joy_act, Y=sad_act, batch_size=57344)
torch.cuda.empty_cache()
# %%
from plot_utils import plot_results


plot_results(results, plot_mean=False)
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

def group_results(results):
    layer_results = defaultdict(lambda: defaultdict(list))
    for layer_idx, size, train_acc, test_acc, train_loss, test_loss in results:
        layer_results[layer_idx][size] = (train_acc, test_acc, train_loss, test_loss)
    return layer_results

def get_stats(layer_results, sample_sizes):
    my_stats = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
    for size in sample_sizes:
        for stat in my_stats:
            values = [layer_data[size][['train_acc', 'test_acc', 'train_loss', 'test_loss'].index(stat)] 
                      for layer_data in layer_results.values() if size in layer_data]
            my_stats[stat].append((np.mean(values), stats.sem(values)))
    return my_stats

def plot_mean_results(ax1, ax2, layer_results, sample_sizes):
    my_stats = get_stats(layer_results, sample_sizes)
    for ax, stat1, stat2 in [(ax1, 'train_acc', 'test_acc'), (ax2, 'train_loss', 'test_loss')]:
        for stat, color, label in [(stat1, 'blue', 'Train'), (stat2, 'orange', 'Test')]:
            means, sems = zip(*my_stats[stat])
            plot_with_confidence(ax, sample_sizes, means, sems, color, label)

def plot_with_confidence(ax, x, means, sems, color, label):
    ax.semilogx(x, means, color=color, label=label)
    ax.fill_between(x, 
                    np.array(means) - 1.96 * np.array(sems),
                    np.array(means) + 1.96 * np.array(sems),
                    alpha=0.3, color=color)

def plot_layer_results(ax1, ax2, layer_results):
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_results)))
    for (layer_idx, layer_data), color in zip(layer_results.items(), colors):
        sizes = sorted(layer_data.keys())
        train_acc, test_acc, train_loss, test_loss = zip(*[layer_data[size] for size in sizes])
        
        for ax, train_data, test_data in [(ax1, train_acc, test_acc), (ax2, train_loss, test_loss)]:
            ax.semilogx(sizes, train_data, color=color, alpha=0.7, linestyle='-', label=f'Layer {layer_idx} (Train)')
            ax.semilogx(sizes, test_data, color=color, alpha=0.7, linestyle='--', label=f'Layer {layer_idx} (Test)')

def set_axis_properties(ax, ylabel, plot_mean):
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs. Training Set Size' + (' (Layer-wise Mean)' if plot_mean else ' for All Layers'))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    if ylabel == 'Loss':
        ax.set_yscale('log')

def print_final_accuracies(layer_results):
    print("Final test accuracies for each layer:")
    for layer_idx, layer_data in layer_results.items():
        final_size = max(layer_data.keys())
        final_test_acc = layer_data[final_size][1]  # test_acc is at index 1
        print(f"Layer {layer_idx}: {final_test_acc:.4f}")
        
def plot_results(results, plot_mean=False):
    layer_results = group_results(results)
    sample_sizes = sorted(set(size for _, size, *_ in results))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    if plot_mean:
        plot_mean_results(ax1, ax2, layer_results, sample_sizes)
    else:
        plot_layer_results(ax1, ax2, layer_results)
    
    set_axis_properties(ax1, 'Accuracy', plot_mean)
    set_axis_properties(ax2, 'Loss', plot_mean)
    
    plt.tight_layout()
    plt.show()
    
    print_final_accuracies(layer_results)
        results.pkl.backup - first set of results in the google doc
results.pkl.backup-new - second set of results in the google doc, run till 1e-4 loss
results.pkl- third set of results in the google doc, run everything for 100 epochs
���)      ]�((K M�G?�E�   G?�gȠ   G?׶b�   G?�I��   t�(KM�G?�"�    G?��@    G?�J_    G?��    t�(KM�G?���   G?�Y��   G?ժ�`   G?�3��   t�(KM�G?����   G?�~�    G?ӽ    G?��6    t�(KM�G?����   G?�۽�   G?թ��   G?�r�    t�(KM�G?��
`   G?����   G?֝A�   G?�e��   t�(KM�G?��`   G?����   G?�L]�   G?�|0    t�(KM�G?�/�   G?�%r�   G?��H    G?���@   t�(KM�G?�      G?��|    G?��C@   G?��    t�(K	M�G?�I��   G?���    G?ӭ|�   G?�,�`   t�(K
M�G?홙�   G?���   G?�[��   G?�1��   t�(KM�G?�/�   G?�71@   G?�i    G?�5��   t�(K M'G?�X�@   G?����   G?��\    G?�*^    t�(KM'G?鞑�   G?���   G?۟�`   G?�I�   t�(KM'G?�k�   G?�tҀ   G?ْ�`   G?�N�    t�(KM'G?���   G?巾�   G?�{��   G?�߂�   t�(KM'G?�h΀   G?�-2    G?ԍ_�   G?�<B�   t�(KM'G?�     G?�S�   G?���   G?���   t�(KM'G?��;�   G?�P4    G?�c�   G?���   t�(KM'G?�/�    G?�j@   G?�[3�   G?��   t�(KM'G?��9@   G?�G��   G?��ۀ   G?�"�@   t�(K	M'G?�v�@   G?�M�    G?՛r�   G?�<u�   t�(K
M'G?�b��   G?�`    G?�<��   G?�fO�   t�(KM'G?칌    G?�`�   G?�Z��   G?��    t�(K M�G?����   G?����   G?����   G?���   t�(KM�G?����   G?�]�`   G?�    G?�`   t�(KM�G?�Q6�   G?���   G?�[��   G?�#s@   t�(KM�G?�K`   G?�U�@   G?��@   G?���   t�(KM�G?�~�`   G?悉�   G?�*��   G?�Ku    t�(KM�G?�`   G?��_`   G?ڊ�@   G?�K��   t�(KM�G?�1o@   G?�z@   G?Ӥ�   G?�f�    t�(KM�G?��I�   G?癌    G?�p>`   G?ᳳ    t�(KM�G?��    G?���   G?��`   G?��;    t�(K	M�G?�-�   G?�N�   G?�5�   G?�u`   t�(K
M�G?�ܰ�   G?��7�   G?�h&@   G?��k    t�(KM�G?�D��   G?�\��   G?�I�   G?�@�   t�(K M�G?�w\�   G?偪�   G?��    G?㯚`   t�(KM�G?��    G?��@   G?����   G?�*��   t�(KM�G?�_-�   G?�u[    G?�y�   G?��*�   t�(KM�G?�?�   G?�pt�   G?�B+�   G?�B    t�(KM�G?�2`   G?��q�   G?��`   G?�ï    t�(KM�G?��ހ   G?��p�   G?�\�   G?�d{�   t�(KM�G?�F�   G?�~V�   G?�3�    G?�fd    t�(KM�G?�c�`   G?��^@   G?�"�`   G?�Q7`   t�(KM�G?�	��   G?��!�   G?���   G?�s@`   t�(K	M�G?�V�   G?���    G?ץ�   G?�"    t�(K
M�G?�Cm`   G?��'�   G?�P��   G?�d    t�(KM�G?��/�   G?祋�   G?�c�@   G?�'�   t�(K M�G?�8׀   G?� �   G?�nj�   G?�P�    t�(KM�G?���   G?��   G?��~�   G?���    t�(KM�G?�i}�   G?挂    G?܀�   G?�,    t�(KM�G?��$@   G?���    G?�2<�   G?�	�    t�(KM�G?�0��   G?��B`   G?�T�   G?�ڐ�   t�(KM�G?����   G?�O�@   G?����   G?�    t�(KM�G?��$@   G?��À   G?����   G?��`   t�(KM�G?���   G?��L    G?�)b`   G?��9�   t�(KM�G?��a�   G?��@   G?�A&@   G?�E4`   t�(K	M�G?���   G?�Pf    G?ٳ�    G?�[�   t�(K
M�G?��    G?�F    G?�*7    G?���   t�(KM�G?�@    G?���   G?��L�   G?�ě�   t�(K M�G?��-�   G?�    G?�t@   G?��J    t�(KM�G?�9@   G?�/9    G?�Jo    G?�`�`   t�(KM�G?���   G?���    G?��*@   G?��H    t�(KM�G?�b�   G?�)�    G?�H`   G?�4�   t�(KM�G?�:��   G?�IF`   G?���@   G?�^[    t�(KM�G?��F�   G?�C�   G?ڗ�    G?�2    t�(KM�G?�"p�   G?�-�@   G?���   G?�0�    t�(KM�G?�;�   G?�J�@   G?ڈ�   G?�U`   t�(KM�G?���   G?�2�    G?�O�@   G?��z@   t�(K	M�G?��    G?�[�   G?�I�@   G?�c�    t�(K
M�G?�	ˀ   G?���   G?�W�`   G?��   t�(KM�G?��s�   G?�4-@   G?���`   G?�|�`   t�(K M�G?���    G?����   G?���   G?◃�   t�(KM�G?��@   G?�yi`   G?�F��   G?�"�   t�(KM�G?�c�   G?���   G?�!@�   G?�j�   t�(KM�G?���   G?�6��   G?�R�    G?�I@   t�(KM�G?�q��   G?���   G?��B    G?�),    t�(KM�G?�8k�   G?�x�`   G?�>w�   G?�)*�   t�(KM�G?驾�   G?�D;@   G?���@   G?��   t�(KM�G?�1`   G?�jA@   G?�V�`   G?��[�   t�(KM�G?�NL�   G?�    G?�w�    G?�VY`   t�(K	M�G?�z    G?�P�   G?�K��   G?��     t�(K
M�G?�-��   G?�N_    G?�'݀   G?��`   t�(KM�G?�2�   G?秾    G?�ʵ�   G?�}    t�(K M2G?�;l    G?惸`   G?�p�@   G?�G@   t�(KM2G?�L�   G?林    G?����   G?���@   t�(KM2G?�־@   G?�n��   G?�]j`   G?�#��   t�(KM2G?��   G?�J�`   G?��K�   G?�E�    t�(KM2G?�.�   G?���   G?�%�   G?��@   t�(KM2G?�p�    G?�5�   G?�=?    G?�PY�   t�(KM2G?�q�   G?豽`   G?���   G?���    t�(KM2G?�q�   G?����   G?�؈@   G?�`8�   t�(KM2G?�o��   G?��   G?ܾp�   G?�l`   t�(K	M2G?��_�   G?�v@   G?ۚ�   G?��`   t�(K
M2G?��@   G?�;F@   G?޻�   G?�*��   t�(KM2G?�tj�   G?���   G?���@   G?��P@   t�(K M�#G?�K��   G?�S8    G?�L�`   G?�b�   t�(KM�#G?�&    G?���    G?��   G?��    t�(KM�#G?��    G?�8    G?��   G?���   t�(KM�#G?���   G?�:��   G?�t��   G?�N�@   t�(KM�#G?�	��   G?��z�   G?���@   G?�Q�   t�(KM�#G?�H@   G?��!    G?����   G?��@@   t�(KM�#G?�?�@   G?�~]�   G?�� �   G?߃5@   t�(KM�#G?�    G?�n    G?���   G?��   t�(KM�#G?�6 �   G?��    G?���    G?��!`   t�(K	M�#G?�$@�   G?�d+�   G?ݲ3    G?�R��   t�(K
M�#G?��@   G?���    G?ۉ�   G?�=x`   t�(KM�#G?�#\�   G?�xs`   G?�e�    G?��2�   t�(K MZ/G?��    G?���@   G?ᝦ�   G?�a �   t�(KMZ/G?縴    G?�ͽ`   G?�ܖ    G?��    t�(KMZ/G?�=(�   G?狅@   G?��    G?� -�   t�(KMZ/G?��A�   G?�4�@   G?�`��   G?�B    t�(KMZ/G?轏�   G?���`   G?ޣ��   G?�rJ@   t�(KMZ/G?�>�`   G?�JP�   G?��`   G?�Ԡ�   t�(KMZ/G?�gܠ   G?���   G?�x��   G?޺G�   t�(KMZ/G?�:    G?����   G?�U��   G?�(5@   t�(KMZ/G?��s@   G?��    G?�߿�   G?��   t�(K	MZ/G?�_�   G?蹮�   G?ܫi    G?� 5�   t�(K
MZ/G?�U��   G?�y`   G?ܡ    G?��v�   t�(KMZ/G?��    G?�j    G?�ot�   G?�\`   t�(K M{>G?�o�   G?漀�   G?�'�@   G?��    t�(KM{>G?�{9    G?���   G?���   G?�!�   t�(KM{>G?�{@   G?�[�@   G?��@   G?�=�    t�(KM{>G?�Y�   G?�u�   G?߫*`   G?�C�   t�(KM{>G?���   G?�     G?�n��   G?�*�`   t�(KM{>G?��>�   G?�F    G?����   G?�U    t�(KM{>G?�/    G?��    G?�z�    G?�ۀ   t�(KM{>G?��G�   G?�	��   G?�.[�   G?�܉�   t�(KM{>G?�1�@   G?�Oi@   G?�/��   G?�/��   t�(K	M{>G?���   G?�`    G?�xo    G?ߒ�@   t�(K
M{>G?�o�`   G?�ռ@   G?܌T    G?�l`   t�(KM{>G?�Y@   G?�?)`   G?�(	    G?�a�    t�(K MqRG?���   G?�4�    G?�ީ`   G?�3�@   t�(KMqRG?�Jp�   G?����   G?�K��   G?ខ`   t�(KMqRG?��@   G?���   G?�^��   G?���    t�(KMqRG?�js�   G?��E@   G?ߩI    G?�w�    t�(KMqRG?臐    G?��@   G?�`�    G?�A'@   t�(KMqRG?�^�   G?�]�   G?ߠ��   G?�H�    t�(KMqRG?�ú    G?�l�    G?ޤq�   G?߇#�   t�(KMqRG?�/?    G?�؛�   G?�;�`   G?�Z�    t�(KMqRG?���   G?��    G?�l�   G?ߎT    t�(K	MqRG?�j�    G?��    G?�t��   G?��    t�(K
MqRG?艁    G?�7c@   G?�     G?��v    t�(KMqRG?��`   G?��9�   G?�Rz�   G?�z�   t�(K M�lG?��    G?��   G?��   G?�ޱ    t�(KM�lG?�	��   G?���@   G?���   G?��'�   t�(KM�lG?�B��   G?��2�   G?�'�@   G?�w}    t�(KM�lG?�@�@   G?���   G?��(`   G?�R(    t�(KM�lG?�(��   G?��>    G?�#    G?�s׀   t�(KM�lG?��    G?����   G?�E��   G?��    t�(KM�lG?�hK�   G?�g�   G?�xX�   G?ݐ��   t�(KM�lG?��b�   G?�֔�   G?��     G?ބ�    t�(KM�lG?��    G?�N�    G?�`   G?��<@   t�(K	M�lG?�m`   G?�g6�   G?�G�    G?� &�   t�(K
M�lG?�)    G?���   G?ݣ*�   G?ހ%    t�(KM�lG?��&�   G?���    G?��~    G?�/�@   t�(K M��G?�A��   G?��Q    G?�UF�   G?�:�   t�(KM��G?�*!�   G?�� �   G?�``   G?�)    t�(KM��G?���   G?��    G?ෙ    G?��    t�(KM��G?�*    G?���   G?�|    G?�|     t�(KM��G?��    G?���   G?�(�    G?�y�    t�(KM��G?�]\�   G?��   G?ߘ�`   G?�0>�   t�(KM��G?�%�    G?��@   G?�[�    G?�*h�   t�(KM��G?�'@   G?�GL�   G?���    G?����   t�(KM��G?�Y�    G?�d    G?��7    G?�۵    t�(K	M��G?�c(�   G?�Qi�   G?��`   G?�xP    t�(K
M��G?�N��   G?���   G?����   G?���   t�(KM��G?�|�   G?�z�    G?��V    G?���`   t�(K Mb�G?��    G?�3    G?���    G?�b�   t�(KMb�G?�F��   G?����   G?�SO�   G?��    t�(KMb�G?�K�@   G?��@   G?�	O    G?�^;@   t�(KMb�G?��`   G?�⍀   G?�,g    G?��|�   t�(KMb�G?荮`   G?�4�    G?�&��   G?��V�   t�(KMb�G?�zk@   G?�C�   G?�\@   G?��`   t�(KMb�G?�    G?��6    G?݈��   G?�D��   t�(KMb�G?��   G?�P�   G?��O�   G?����   t�(KMb�G?��o`   G?�n    G?�q��   G?�2��   t�(K	Mb�G?��\�   G?��@   G?�~��   G?�Cy�   t�(K
Mb�G?�Z�   G?�g�`   G?�$�`   G?���   t�(KMb�G?���   G?�D    G?�C��   G?��o    t�(K M��G?���@   G?�y�   G?�i`   G?�Л�   t�(KM��G?�N��   G?�$��   G?�:؀   G?�gb    t�(KM��G?�>O    G?�5�   G?�]��   G?ძ    t�(KM��G?礀@   G?�ow�   G?���   G?��c@   t�(KM��G?�;�   G?炻�   G?��@   G?��N�   t�(KM��G?�h    G?�Z�`   G?���    G?߃�    t�(KM��G?�W�    G?�1�    G?ܧq@   G?�<    t�(KM��G?�m`   G?虾    G?ުg�   G?�:�   t�(KM��G?�X�   G?�6��   G?��j    G?݅�   t�(K	M��G?�\ŀ   G?�/�    G?�ԟ    G?�}'@   t�(K
M��G?�J�   G?���   G?�y�   G?�l�    t�(KM��G?�&`   G?� R`   G?�~,�   G?�W`   t�(K J�I G?��[    G?�<`   G?���   G?��Y�   t�(KJ�I G?�7�    G?�S@   G?�l�`   G?��   t�(KJ�I G?�B؀   G?� �`   G?�&�@   G?�L��   t�(KJ�I G?礦    G?�/    G?�ɛ�   G?���   t�(KJ�I G?�06    G?�N�   G?�À   G?�B��   t�(KJ�I G?�3̀   G?�'��   G?��    G?�(,    t�(KJ�I G?�]�   G?輎    G?�]�`   G?ޙ��   t�(KJ�I G?�_`   G?�
��   G?�n�`   G?��b�   t�(KJ�I G?�h`   G?�TO�   G?ܭ	    G?�%    t�(K	J�I G?��   G?���@   G?���`   G?�BI�   t�(K
J�I G?軭�   G?�&    G?�;��   G?�r�   t�(KJ�I G?踒`   G?�T�   G?��    G?��@   t�(K J� G?�ۀ   G?�V�   G?��`   G?៮�   t�(KJ� G?��w@   G?��    G?��#�   G?��|`   t�(KJ� G?矤�   G?�?�   G?��t    G?���   t�(KJ� G?���   G?�8    G?��   G?�g`   t�(KJ� G?��y    G?��B    G?���@   G?����   t�(KJ� G?�l�    G?�]>    G?߆��   G?ߪp@   t�(KJ� G?�mF�   G?�Rʀ   G?�`U�   G?ܫ��   t�(KJ� G?�=�@   G?�/M`   G?��`   G?�Xh`   t�(KJ� G?�2H�   G?�%��   G?�;�   G?ݐ��   t�(K	J� G?�]�@   G?�\e�   G?�*�   G?�C�@   t�(K
J� G?�2��   G?�;�    G?ौ�   G?ࣩ�   t�(KJ� G?�@    G?��   G?�$V    G?�7��   t�(K J> G?��U�   G?���`   G?ᨇ�   G?ᦀ�   t�(KJ> G?�+�   G?�=    G?�
r�   G?��@   t�(KJ> G?�C    G?�"�   G?��d    G?��%`   t�(KJ> G?�4f�   G?�#�    G?�'W`   G?�2��   t�(KJ> G?���   G?��ՠ   G?�q@   G?���    t�(KJ> G?��0@   G?��s`   G?�[�    G?�]L@   t�(KJ> G?�H-`   G?�C�    G?ܺ�    G?��g�   t�(KJ> G?���   G?���    G?��4@   G?��P�   t�(KJ> G?�8�   G?�5`   G?�;�   G?�dm`   t�(K	J> G?�wZ@   G?�j��   G?��à   G?�L�   t�(K
J> G?�Y�   G?�L�`   G?�{�   G?�c    t�(KJ> G?�	k�   G?�
��   G?�É�   G?�Ͽ`   t�(K Jk� G?�m�    G?�o�    G?�!��   G?��    t�(KJk� G?�*�    G?���   G?�y^�   G?�v@�   t�(KJk� G?��    G?��@   G?���   G?��1�   t�(KJk� G?�G�    G?�0    G?���   G?��    t�(KJk� G?����   G?��3�   G?�y��   G?�_�   t�(KJk� G?�.    G?谹�   G?ޟ3`   G?��    t�(KJk� G?�D�   G?���   G?�b��   G?݉x`   t�(KJk� G?��   G?��{�   G?޴$�   G?�Ƒ�   t�(KJk� G?�QM�   G?�K��   G?���   G?�
S    t�(K	Jk� G?�2�   G?�4�`   G?�`�    G?�zҀ   t�(K
Jk� G?�ꮀ   G?��w�   G?މ��   G?���    t�(KJk� G?软�   G?��   G?�w�   G?�.?@   t�e.import os
import pickle
import torch

def cache_function_call(file_path, func, use_torch=False, *args, **kwargs):
    # Check if the results file exists
    if os.path.exists(file_path):
        # Load the results from the file
        if use_torch:
            results = torch.load(file_path, map_location=torch.device('cpu'))
            print(f"Results loaded from disk using torch.load: {file_path}")
        else:
            with open(file_path, 'rb') as file:
                results = pickle.load(file)
            print(f"Results loaded from disk using pickle: {file_path}")
    else:
        # Run the function to get the results
        results = func(*args, **kwargs)
        
        # Move results to CPU if it's a torch tensor
        if use_torch and isinstance(results, torch.Tensor):
            results = results.cpu()
        
        # Save the results to the file
        if use_torch:
            torch.save(results, file_path)
            print(f"Results computed and saved to disk using torch.save: {file_path}")
        else:
            with open(file_path, 'wb') as file:
                pickle.dump(results, file)
            print(f"Results computed and saved to disk using pickle: {file_path}")
        
    return results