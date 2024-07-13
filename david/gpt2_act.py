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
