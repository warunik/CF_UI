import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_counterfactual_comparison(original: np.ndarray,
                                   raw_cfs: List[np.ndarray],
                                   feasible_cfs: List[np.ndarray],
                                   feature_names: List[str] = None,
                                   highlight_changes: bool = True):
    """
    Visualize comparison between original, raw CFs, and feasible CFs
    
    Args:
        original: Original instance (n_features,)
        raw_cfs: List of raw counterfactuals
        feasible_cfs: List of feasible counterfactuals
        feature_names: Names of features
        highlight_changes: Whether to highlight changed features
    """
    n_features = original.shape[0]
    n_cfs = len(raw_cfs)
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(n_features)]
    
    fig, axes = plt.subplots(n_cfs, 1, figsize=(10, 3 * n_cfs), squeeze=False)
    axes = axes.flatten()
    
    for i in range(n_cfs):
        ax = axes[i]
        index = np.arange(n_features)
        bar_width = 0.25
        
        # Plot values
        ax.bar(index - bar_width, original, bar_width, label='Original')
        ax.bar(index, raw_cfs[i], bar_width, label='Raw CF', alpha=0.7)
        ax.bar(index + bar_width, feasible_cfs[i], bar_width, label='Feasible CF')
        
        # Highlight changes if requested
        if highlight_changes:
            changed_indices = np.where(np.abs(raw_cfs[i] - feasible_cfs[i]) > 1e-5)[0]
            for idx in changed_indices:
                ax.patches[2 * n_features + idx].set_facecolor('r')
                ax.patches[n_features + idx].set_facecolor('orange')
        
        ax.set_title(f'Counterfactual {i+1}')
        ax.set_xticks(index)
        ax.set_xticklabels(feature_names, rotation=45)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig

def plot_feasibility_scores(scores: List[float], labels: List[str] = None):
    """Plot feasibility scores for multiple counterfactuals"""
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(scores))
    
    ax.bar(x, scores, color='skyblue')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Counterfactual Index')
    ax.set_ylabel('Feasibility Score')
    ax.set_title('Feasibility Scores Comparison')
    
    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    
    for i, score in enumerate(scores):
        ax.text(i, score + 0.02, f"{score:.2f}", ha='center')
    
    return fig

def plot_density_comparison(original: np.ndarray, 
                            counterfactuals: List[np.ndarray],
                            density_validator):
    """Plot density comparison between instances"""
    instances = [original] + counterfactuals
    densities = [density_validator.score(inst) for inst in instances]
    labels = ['Original'] + [f'CF {i+1}' for i in range(len(counterfactuals))]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, densities, color=['blue'] + ['green']*len(counterfactuals))
    ax.axhline(y=density_validator.threshold, color='r', linestyle='--', 
               label='Density Threshold')
    ax.set_ylabel('Log Density')
    ax.set_title('Density Comparison')
    ax.legend()
    plt.xticks(rotation=45)
    return fig