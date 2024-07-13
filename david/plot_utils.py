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
        