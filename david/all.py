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
Äï•)      ]î((K MËG?Î∂E¿   G?‰g»†   G?◊∂b†   G?ÊIø†   tî(KMËG?Î"—    G?‰“@    G?ŸJ_    G?‰≤ﬁ    tî(KMËG?ÏìÄ   G?ÂYÛ¿   G?’™Î`   G?Â3õ†   tî(KMËG?ÏÃÃ‡   G?Â~À    G?”Ω    G?ÂÈ6    tî(KMËG?Î˜Œ‡   G?Â€ΩÄ   G?’©—‡   G?‰r’    tî(KMËG?Î◊
`   G?Âı√‡   G?÷ùA†   G?‰e˚‡   tî(KMËG?ÌÛ∂`   G?Ê˘≠†   G?ŒL]¿   G?‰|0    tî(KMËG?Ì/¿   G?Á%r†   G?–ÈH    G?„¿‰@   tî(KMËG?Ï      G?Êﬂ|    G?‘ÍC@   G?„Ì    tî(K	MËG?ÏI∫Ä   G?Êˆ£    G?”≠|Ä   G?‰,ë`   tî(K
MËG?Ìôô¿   G?Ê‚Ü¿   G?Ã[—‡   G?Ê1§†   tî(KMËG?Ì/¿   G?Á71@   G?—i    G?Â5ºÄ   tî(K M'G?ÍX‰@   G?‰ÙπÄ   G?Ÿ\    G?Â*^    tî(KM'G?Èûë†   G?Â√‡   G?€üò`   G?‰I‡   tî(KM'G?Í£k¿   G?Ât“Ä   G?ŸíÜ`   G?‰N⁄    tî(KM'G?Íñˇ‡   G?Â∑æÄ   G?Ÿ{‹¿   G?„ﬂÇ‡   tî(KM'G?ÏhŒÄ   G?Ê-2    G?‘ç_‡   G?‰<B¿   tî(KM'G?Ï≠     G?ÊS†   G?“®†   G?‰ªÛ¿   tî(KM'G?ÏÎ;†   G?ÁP4    G?—c¿   G?„Î‡   tî(KM'G?Ì/ç    G?Áj@   G?œ[3‡   G?„â‡   tî(KM'G?Îˇ9@   G?ÁGï¿   G?”€€Ä   G?„"ö@   tî(K	M'G?Îvñ@   G?ÁM´    G?’õr¿   G?„<uÄ   tî(K
M'G?Ïbò†   G?Á±`    G?—<ΩÄ   G?„fOÄ   tî(KM'G?Ïπå    G?Áô`¿   G?—Zü‡   G?‰éÓ    tî(K MÃG?Ë·¸Ä   G?‰¯«‡   G?›ﬂ˜‡   G?‰ª‡   tî(KMÃG?Ë·¸Ä   G?Â]Ä`   G?ﬁ    G?„ñ`   tî(KMÃG?ÍQ6¿   G?Êﬂ¿   G?€[Ä‡   G?„#s@   tî(KMÃG?ÍÄK`   G?ÊUÏ@   G?⁄ó@   G?„±¿   tî(KMÃG?Î~á`   G?ÊÇâ†   G?◊*Œ‡   G?„Ku    tî(KMÃG?Í`   G?Ê◊_`   G?⁄ä›@   G?‚Ká†   tî(KMÃG?Ï1o@   G?Áz@   G?”§†   G?‚fÊ    tî(KMÃG?ÍÒIÄ   G?Áôå    G?◊p>`   G?·≥≥    tî(KMÃG?Ïö    G?Á°”‡   G?”±`   G?‚¬;    tî(K	MÃG?ÏÜ-‡   G?ÁîN‡   G?—5Í†   G?„†u`   tî(K
MÃG?Î‹∞Ä   G?Á 7†   G?‘h&@   G?‚·k    tî(KMÃG?ÈD⁄‡   G?Á\µÄ   G?‹IÌÄ   G?‰@Ä   tî(K M˘G?Èw\‡   G?ÂÅ™Ä   G?‹‰≥    G?„Øö`   tî(KM˘G?Ë≈    G?ÂáÎ@   G?ﬁ˛å†   G?„*Ñ¿   tî(KM˘G?Í_-†   G?Êu[    G?⁄y¿   G?‚„*¿   tî(KM˘G?Í?†   G?ÊptÄ   G?⁄B+Ä   G?‚∏B    tî(KM˘G?Î2`   G?Ê–q†   G?ÿ—`   G?‚√Ø    tî(KM˘G?Í‘ﬁÄ   G?ÊÙpÄ   G?ÿ\ÚÄ   G?‚d{Ä   tî(KM˘G?ÍF†   G?Á~V¿   G?⁄3◊    G?·fd    tî(KM˘G?ÎcÜ`   G?ÁÌ^@   G?÷"¥`   G?·Q7`   tî(KM˘G?Í	ï¿   G?Á–!‡   G?Ÿ¯‡   G?·s@`   tî(K	M˘G?Í∏VÄ   G?ÁÊ«    G?◊•¿   G?·°"    tî(K
M˘G?ÎCm`   G?Áˆ'‡   G?÷P∏¿   G?‚d    tî(KM˘G?ÈÙ/¿   G?Á•ã†   G?€c•@   G?„'†   tî(K M◊G?È8◊Ä   G?Âé †   G?ﬁnjÄ   G?„P¶    tî(KM◊G?Ëì˘Ä   G?Âµ‚†   G?ﬁ“~Ä   G?‚≈‰    tî(KM◊G?Èi}¿   G?ÊåÇ    G?‹Ä‡   G?‚,    tî(KM◊G?È˛$@   G?Ê”˛    G?€2<¿   G?‚	¶    tî(KM◊G?È0ª¿   G?Ê€B`   G?›T†   G?·⁄ê¿   tî(KM◊G?È¯ºÄ   G?ÁO≤@   G?⁄Ê¡Ä   G?·å    tî(KM◊G?È˛$@   G?ÁÂ√Ä   G?⁄Ìœ¿   G?‡ø‹`   tî(KM◊G?È€Ä   G?ÁÙL    G?€)b`   G?‡ü9‡   tî(KM◊G?Ë˝a†   G?Á»@   G?›A&@   G?·E4`   tî(K	M◊G?ÍÛ‡   G?ËPf    G?Ÿ≥∑    G?‡[Ä   tî(K
M◊G?Í˘    G?ËF    G?◊*7    G?·›¿   tî(KM◊G?Í@    G?Ë°Ä   G?⁄‚L¿   G?·ƒõ†   tî(K MüG?Á»-†   G?Â∏    G?‡t@   G?‚–J    tî(KMüG?Ëù9@   G?Ê/9    G?ﬂJo    G?‚`Ø`   tî(KMüG?ÈéÚ¿   G?ÊÁÔ    G?‹Œ*@   G?·‚H    tî(KMüG?È°bÄ   G?Á)Å    G?‹H`   G?·≤4†   tî(KMüG?È:ıÄ   G?ÁIF`   G?‹À’@   G?·^[    tî(KMüG?ÈÒF‡   G?Á©CÄ   G?⁄ó®    G?·2    tî(KMüG?Í"p‡   G?Ë-¡@   G?⁄Ô‡   G?‡0ã    tî(KMüG?Í;‡   G?ËJß@   G?⁄àÈ†   G?‡U`   tî(KMüG?È¿¿   G?Ë2”    G?€Oñ@   G?‡íz@   tî(K	MüG?Í√    G?Ë[†   G?⁄I‡@   G?‡c◊    tî(K
MüG?È	ÀÄ   G?Á¿‡   G?ﬁW `   G?‚¿   tî(KMüG?ÈŒs¿   G?Ë4-@   G?€≈ﬁ`   G?·|ì`   tî(K MùG?ÁÛ¯    G?ÂˆÚ¿   G?‡áÄ   G?‚óÉÄ   tî(KMùG?Ë‚@   G?Êyi`   G?‡FÙ‡   G?‚"†   tî(KMùG?Ë£c‡   G?Á≥¿   G?ﬂ!@†   G?·èj‡   tî(KMùG?È—‡   G?Á6⁄¿   G?ﬁRˇ    G?·ÇI@   tî(KMùG?Èq€‡   G?Áñ◊‡   G?‹˛B    G?·),    tî(KMùG?È8k†   G?ÁxÓ`   G?ﬁ>wÄ   G?·)*†   tî(KMùG?È©æ¿   G?ËD;@   G?€˘®@   G?‡¿   tî(KMùG?È®1`   G?ËjA@   G?€Vƒ`   G?ﬂﬂ[Ä   tî(KMùG?ÍNL†   G?Ë≠    G?ŸwÊ    G?ﬂVY`   tî(K	MùG?Èz    G?ËP†   G?›K∞‡   G?‡ô     tî(K
MùG?È-ç†   G?ËN_    G?›'›Ä   G?‡™–`   tî(KMùG?Ë2†   G?Áßæ    G?· µ†   G?„ø}    tî(K M2G?Ë;l    G?ÊÉ∏`   G?‡pä@   G?‚G@   tî(KM2G?ËLÄ   G?Êûó    G?‡Äÿ¿   G?·¯î@   tî(KM2G?Ë÷æ@   G?Ánı‡   G?ﬁ]j`   G?·#ﬁ¿   tî(KM2G?Ë¶‡   G?ÁJ†`   G?ﬁ‚K†   G?·Eß    tî(KM2G?Ë.†   G?Áû†   G?‡%·†   G?·ÑŒ@   tî(KM2G?Èp„    G?Ë5Ä   G?‹=?    G?‡PY†   tî(KM2G?Íq‡   G?Ë±Ω`   G?⁄∆¿   G?ﬁ¯Å    tî(KM2G?Íq‡   G?ËÂıÄ   G?Ÿÿà@   G?ﬁ`8¿   tî(KM2G?Èoµ‡   G?ËÉ‡   G?‹æp¿   G?‡l`   tî(K	M2G?È—_‡   G?Ëëv@   G?€öÄ   G?‡≠`   tî(K
M2G?Ë°@   G?Ë;F@   G?ﬁªÄ   G?·*ı¿   tî(KM2G?Ètj‡   G?ËòÂ¿   G?‹Î¸@   G?‡üP@   tî(K M„#G?ÁKß†   G?ÊS8    G?·L™`   G?‚b¿   tî(KM„#G?Á¢&    G?Ê≈Ã    G?‡ÒºÄ   G?·ƒ    tî(KM„#G?Ëº—    G?Áæ8    G?ﬂ¿   G?‡¿†   tî(KM„#G?ÁÌÒÄ   G?Á:íÄ   G?‡tÎ¿   G?·N¯@   tî(KM„#G?È	ÄÄ   G?Áız‡   G?›…ˆ@   G?‡Q†   tî(KM„#G?Ë†H@   G?ÁË!    G?ﬁÔÂ¿   G?‡ç@@   tî(KM„#G?È?Â@   G?Ë~]Ä   G?‹Ú ¿   G?ﬂÉ5@   tî(KM„#G?È°    G?Ë≥n    G?‹ˇ‡   G?ﬂÚ†   tî(KM„#G?Í6 ‡   G?Èñ    G?Ÿˇª    G?› !`   tî(K	M„#G?È$@¿   G?Ëd+‡   G?›≤3    G?‡RÇ‡   tî(K
M„#G?È¬@   G?Ë€˝    G?€â†   G?ﬂ=x`   tî(KM„#G?È#\Ä   G?Ëxs`   G?ﬁeâ    G?‡Ù2Ä   tî(K MZ/G?Êˆ    G?Â˝µ@   G?·ù¶†   G?‚a ‡   tî(KMZ/G?Á∏¥    G?ÊÕΩ`   G?‡‹ñ    G?·∞Ó    tî(KMZ/G?Ë=(‡   G?ÁãÖ@   G?‡’    G?· -¿   tî(KMZ/G?Á˙AÄ   G?Á4®@   G?‡`°¿   G?·B    tî(KMZ/G?ËΩè†   G?Á·‡`   G?ﬁ£ ‡   G?‡rJ@   tî(KMZ/G?È>£`   G?ËJP†   G?›Ï`   G?ﬂ‘††   tî(KMZ/G?Èg‹†   G?Ë≤¿‡   G?‹xôÄ   G?ﬁ∫G¿   tî(KMZ/G?È¥:    G?Ë‚Í¿   G?€Uõ†   G?ﬁ(5@   tî(KMZ/G?È›s@   G?È¨    G?⁄ﬂø‡   G?›Òã†   tî(K	MZ/G?È_Ä   G?ËπÆ†   G?‹´i    G?ﬂ 5Ä   tî(K
MZ/G?ÈUùÄ   G?Ëûy`   G?‹°    G?ﬂœvÄ   tî(KMZ/G?Ë˝    G?Ëj    G?ﬂot‡   G?·\`   tî(K M{>G?ÁoÙÄ   G?ÊºÄ†   G?·'…@   G?·„£    tî(KM{>G?Á{9    G?ÊÂ†   G?·Â¿   G?·§!¿   tî(KM{>G?Ë{@   G?Á[›@   G?‡ã@   G?·=•    tî(KM{>G?ËíY†   G?Á∑uÄ   G?ﬂ´*`   G?‡∂C¿   tî(KM{>G?ËÚ¿   G?Ë     G?ﬁnË‡   G?‡*É`   tî(KM{>G?Ë˜>‡   G?ËF    G?›ÒÛ‡   G?‡U    tî(KM{>G?È≤/    G?ËÚ¢    G?€z≥    G?ﬁ€Ä   tî(KM{>G?ÈÀG†   G?È	ù‡   G?€.[‡   G?›‹â‡   tî(KM{>G?Í1∂@   G?ÈOi@   G?⁄/˛†   G?›/áÄ   tî(K	M{>G?È¥¿   G?Ë£`    G?›xo    G?ﬂí≥@   tî(K
M{>G?Èoö`   G?Ë’º@   G?‹åT    G?ﬂl`   tî(KM{>G?ËßY@   G?Ë?)`   G?‡(	    G?·a÷    tî(K MqRG?Êî»‡   G?Ê4¯    G?·ﬁ©`   G?‚3Ó@   tî(KMqRG?ÁJp‡   G?ÊçÄ   G?·K¨¿   G?·ûÅ`   tî(KMqRG?Ë‹@   G?Á∫÷¿   G?‡^ìÄ   G?‡…‡    tî(KMqRG?Ëjs†   G?ÁÊE@   G?ﬂ©I    G?‡w≈    tî(KMqRG?Ëáê    G?Ëû@   G?ﬂ`Ó    G?‡A'@   tî(KMqRG?Ë^‡   G?Ë]Ä   G?ﬂ†ì¿   G?‡H£    tî(KMqRG?Ë√∫    G?Ëlü    G?ﬁ§qÄ   G?ﬂá#¿   tî(KMqRG?È/?    G?Ëÿõ¿   G?›;‹`   G?ﬁZÒ    tî(KMqRG?ËÂ‚Ä   G?Ëë¯    G?ﬁlÎÄ   G?ﬂéT    tî(K	MqRG?Èj¢    G?È∫    G?‹tº‡   G?›Ùä    tî(K
MqRG?ËâÅ    G?Ë7c@   G?‡     G?‡¿v    tî(KMqRG?ÈÇ`   G?ËÌ9¿   G?ﬁRz¿   G?ﬂzÄ   tî(K M«lG?Á§    G?Ê¶¿   G?·Ü†   G?·ﬁ±    tî(KM«lG?Á	≤‡   G?Ê«”@   G?·¶ﬁ‡   G?·⁄'Ä   tî(KM«lG?ËBˇ¿   G?Á˘2†   G?‡'‰@   G?‡w}    tî(KM«lG?Ë@•@   G?Ëæ¿   G?ﬂÓ(`   G?‡R(    tî(KM«lG?Ë(Ö‡   G?Á‰>    G?‡#    G?‡s◊Ä   tî(KM«lG?Ëî    G?Á‹˘¿   G?‡E¸‡   G?‡ã    tî(KM«lG?ÈhK†   G?ÈgÄ   G?‹xXÄ   G?›êà‡   tî(KM«lG?ËÂb¿   G?Ë÷î†   G?›Ò     G?ﬁÑ‚    tî(KM«lG?ÈàÛ    G?ÈNë    G?‹`   G?‹Ò<@   tî(K	M«lG?Ëím`   G?Ëg6Ä   G?ﬂGî    G?‡ &‡   tî(K
M«lG?È)    G?ËÏå¿   G?›£*Ä   G?ﬁÄ%    tî(KM«lG?ËÁ&†   G?Ë¿Û    G?ﬂﬂ~    G?‡/¥@   tî(K MáèG?ÁA√¿   G?Ê”Q    G?·UF¿   G?·∏:¿   tî(KMáèG?Á*!†   G?Ê˘ †   G?·``   G?·ü)    tî(KMáèG?Á–Ê†   G?Á£€    G?‡∑ô    G?‡¯    tî(KMáèG?Ë*    G?Á‰‡   G?‡|    G?‡|     tî(KMáèG?Ëß    G?Á‘‡   G?‡(‚    G?‡yÉ    tî(KMáèG?Ë]\¿   G?Ë†   G?ﬂòÈ`   G?‡0>†   tî(KMáèG?È%ì    G?Ë‡@   G?›[·    G?ﬁ*h‡   tî(KMáèG?Èí'@   G?ÈGL¿   G?€—ﬂ    G?‹‚–‡   tî(KMáèG?ÈYø    G?Èd    G?‹–7    G?›€µ    tî(K	MáèG?Ëc(†   G?ËQi¿   G?‡Ô`   G?‡xP    tî(K
MáèG?ËNﬁ¿   G?Ë”‡   G?‡Ü€¿   G?·Í‡   tî(KMáèG?ËÉ|¿   G?Ëz—    G?‡ÑV    G?‡Ê `   tî(K MbΩG?ÊŸ    G?ÊÇ3    G?·«¯    G?‚b¿   tî(KMbΩG?ÁFπÄ   G?Ê˛øÄ   G?·SO‡   G?·ó¬    tî(KMbΩG?ËKù@   G?Ëª@   G?‡	O    G?‡^;@   tî(KMbΩG?Ë†`   G?Á‚çÄ   G?‡,g    G?‡É|¿   tî(KMbΩG?ËçÆ`   G?Ë4Ø    G?ﬂ&†‡   G?ﬂÒV†   tî(KMbΩG?Ëzk@   G?ËC‰†   G?ﬂ\@   G?‡Ã`   tî(KMbΩG?È    G?ËÏ6    G?›àû¿   G?ﬁDê‡   tî(KMbΩG?ÈíË†   G?ÈPÓ†   G?€ÙO‡   G?‹“⁄¿   tî(KMbΩG?Ë›o`   G?Ë≥n    G?ﬁqÄ†   G?ﬂ2˘†   tî(K	MbΩG?Ë‰\‡   G?ËΩË@   G?ﬁ~ìÄ   G?ﬂCy‡   tî(K
MbΩG?ËöZ†   G?Ëg∏`   G?‡$å`   G?‡êÄ   tî(KMbΩG?Ë™È¿   G?ËÉD    G?‡Cá†   G?‡ûo    tî(K M‚˘G?Êˆπ@   G?ÊÆy‡   G?·òi`   G?·–õ¿   tî(KM‚˘G?ÁNí†   G?Á$‡   G?·:ÿÄ   G?·gb    tî(KM‚˘G?Á>O    G?Á5‡   G?·]ñÄ   G?·É´    tî(KM‚˘G?Á§Ä@   G?Áow†   G?‡¥»‡   G?‡Îc@   tî(KM‚˘G?ÁÆ;‡   G?ÁÇª†   G?‡™„@   G?‡€NÄ   tî(KM‚˘G?Ëñh    G?ËZ‡`   G?ﬁ˝˝    G?ﬂÉ©    tî(KM‚˘G?ÈW«    G?È1´    G?‹ßq@   G?›<    tî(KM‚˘G?Ë±m`   G?Ëôæ    G?ﬁ™gÄ   G?ﬂ:¿   tî(KM‚˘G?ÈXÄ   G?È6ë¿   G?‹”j    G?›Ö‡   tî(K	M‚˘G?È\≈Ä   G?È/§    G?‹‘ü    G?›}'@   tî(K
M‚˘G?ËJ¿   G?Á˚Ä   G?·y¿   G?·l    tî(KM‚˘G?È&`   G?È R`   G?ﬁ~,†   G?ﬂW`   tî(K J∑I G?ÊÂ[    G?Êµ<`   G?·∏Œ‡   G?·œY†   tî(KJ∑I G?Á7⁄    G?ÁS@   G?·l¨`   G?·ã†   tî(KJ∑I G?ËBÿÄ   G?Ë È`   G?‡&ı@   G?‡Lß†   tî(KJ∑I G?Á§¶    G?Á/    G?‡…õ†   G?‡Ê¢‡   tî(KJ∑I G?Ë06    G?ËN‡   G?‡√Ä   G?‡BßÄ   tî(KJ∑I G?Ë3ÕÄ   G?Ë'´‡   G?‡È    G?‡(,    tî(KJ∑I G?Ëø]†   G?Ëºé    G?ﬁ]≥`   G?ﬁôπ†   tî(KJ∑I G?È_`   G?È
Ã¿   G?›nï`   G?›·bÄ   tî(KJ∑I G?Èh`   G?ÈTO‡   G?‹≠	    G?›%    tî(K	J∑I G?È‡   G?ËÏ„@   G?›„Œ`   G?ﬁBI†   tî(K
J∑I G?Ëª≠Ä   G?Ë´&    G?ﬂ;¡†   G?ﬂrÒ†   tî(KJ∑I G?Ë∏í`   G?Ë∏T†   G?‡∏    G?‡Ë@   tî(K J≥ G?Á€Ä   G?ÁV¿   G?·ü”`   G?·üÆ†   tî(KJ≥ G?Á»w@   G?Á∑Ã    G?‡Ã#Ä   G?‡◊|`   tî(KJ≥ G?Áü§Ä   G?Á®?‡   G?‡Ít    G?‡Â†   tî(KJ≥ G?Á”¿   G?Áæ8    G?‡®Ä   G?‡¥g`   tî(KJ≥ G?Á‹y    G?ÁŸB    G?‡Ñ¸@   G?‡ã´¿   tî(KJ≥ G?Ël§    G?Ë]>    G?ﬂÜ¶¿   G?ﬂ™p@   tî(KJ≥ G?ÈmFÄ   G?ÈR Ä   G?‹`U‡   G?‹´î†   tî(KJ≥ G?È=¬@   G?È/M`   G?›¶`   G?›Xh`   tî(KJ≥ G?È2H¿   G?È%´Ä   G?›;‡   G?›êúÄ   tî(K	J≥ G?Ë]€@   G?Ë\e¿   G?‡*†   G?‡C@   tî(K
J≥ G?Ë2¿¿   G?Ë;»    G?‡•å‡   G?‡£©Ä   tî(KJ≥ G?Ë©@    G?Ë∏‡   G?‡$V    G?‡7‰¿   tî(K J> G?Ê¸U†   G?Êıü`   G?·®á†   G?·¶ÄÄ   tî(KJ> G?Áè+Ä   G?Áè=    G?·
r‡   G?·ß@   tî(KJ> G?Á∫C    G?Á∏"Ä   G?‡÷d    G?‡“%`   tî(KJ> G?Ë4f¿   G?Ë#Ù    G?‡'W`   G?‡2»¿   tî(KJ> G?Á„ë¿   G?Áﬁ’†   G?‡q@   G?‡Å§    tî(KJ> G?Áˆ0@   G?Áˇs`   G?‡[ﬁ    G?‡]L@   tî(KJ> G?ÈH-`   G?ÈC¿    G?‹∫ˆ    G?‹ÍgÄ   tî(KJ> G?Ëı†   G?Ëıÿ    G?›Ê4@   G?›˙P‡   tî(KJ> G?È8¿   G?È5`   G?›;¿   G?›dm`   tî(K	J> G?ËwZ@   G?Ëjó¿   G?ﬂ˘√†   G?‡L†   tî(K
J> G?ËY¿   G?ËLÆ`   G?‡{†   G?‡¢c    tî(KJ> G?Ë	k¿   G?Ë
ö†   G?·√â¿   G?·œø`   tî(K Jkı G?Êmá    G?Êoú    G?‚!Ç¿   G?‚˝    tî(KJkı G?Á*ÿ    G?ÁÙ‡   G?·y^Ä   G?·v@¿   tî(KJkı G?Á£„    G?ÁúÌ@   G?‡ÒÑ¿   G?‡Û1Ä   tî(KJkı G?ËG–    G?Ë0    G?‡´‡   G?‡ä    tî(KJkı G?Á◊˙‡   G?Á’3¿   G?‡y±†   G?‡_‡   tî(KJkı G?Ë∑.    G?Ë∞π¿   G?ﬁü3`   G?ﬁ¿    tî(KJkı G?ÈD¿   G?ÈùÄ   G?›bà†   G?›âx`   tî(KJkı G?Ëº¿   G?Ë√{‡   G?ﬁ¥$†   G?ﬁ∆ëÄ   tî(KJkı G?ÈQM‡   G?ÈK‹¿   G?‹’¿   G?›
S    tî(K	Jkı G?Ë2¿   G?Ë4⁄`   G?‡`Ù    G?‡z“Ä   tî(K
Jkı G?ËÍÆÄ   G?Ë⁄w†   G?ﬁâ°Ä   G?ﬁ«¸    tî(KJkı G?ËΩØ¿   G?Ëµ†   G?‡w†   G?‡.?@   tîe.import os
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