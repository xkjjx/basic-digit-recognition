"""
Comprehensive Weight Visualization for CNN Digit Recognition Model.

A multi-perspective visualization strategy that reveals how the CNN
processes digits from input to output:

1. Convolutional Filter Grids - Visualize learned filters
2. Feature Map Projections - What patterns each filter detects
3. FC Layer Analysis - How features map to classifications
4. Class Template Analysis - What each digit "looks like" to the network
5. Structured Analysis Data - JSON output for programmatic analysis
6. HTML Dashboard with AI Insights - Human-readable analysis
"""

import argparse
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from common import create_cnn_model

WEIGHTS_PATH = "weights/cnn_weights.pth"
OUTPUT_DIR = "visualizations/cnn"


def load_trained_model(weights_path: str):
    """Load the model with trained weights."""
    model = create_cnn_model()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    return model


def get_layer_weights(model):
    """Extract weights from all layers."""
    weights = {}
    layer_idx = 0
    
    for i, layer in enumerate(model):
        if isinstance(layer, torch.nn.Conv2d):
            weights[f'conv{layer_idx + 1}'] = layer.weight.detach().numpy()
            if layer.bias is not None:
                weights[f'conv{layer_idx + 1}_bias'] = layer.bias.detach().numpy()
            layer_idx += 1
        elif isinstance(layer, torch.nn.Linear):
            if 'fc1' not in weights:
                weights['fc1'] = layer.weight.detach().numpy()
                if layer.bias is not None:
                    weights['fc1_bias'] = layer.bias.detach().numpy()
            else:
                weights['fc2'] = layer.weight.detach().numpy()
                if layer.bias is not None:
                    weights['fc2_bias'] = layer.bias.detach().numpy()
    
    return weights


# =============================================================================
# 1. CONVOLUTIONAL FILTER GRIDS
# =============================================================================

def create_conv_filter_grid(filters, title, output_path, grid_shape=None):
    """Create a grid showing all convolutional filters."""
    num_filters = filters.shape[0]
    in_channels = filters.shape[1]
    
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(num_filters)))
        rows = int(np.ceil(num_filters / cols))
    else:
        rows, cols = grid_shape
    
    # For multi-channel filters, show each input channel separately
    if in_channels > 1:
        # Show a subset of filters with all their input channels
        num_to_show = min(16, num_filters)
        fig, axes = plt.subplots(num_to_show, in_channels, 
                                  figsize=(in_channels * 0.8, num_to_show * 0.8))
        fig.suptitle(f"{title}\n(showing {num_to_show}/{num_filters} filters × {in_channels} input channels)", 
                     fontsize=10, fontweight='bold', y=1.02, family='monospace')
        
        vabs = max(abs(filters.min()), abs(filters.max()))
        
        for f_idx in range(num_to_show):
            for c_idx in range(in_channels):
                ax = axes[f_idx, c_idx] if num_to_show > 1 else axes[c_idx]
                kernel = filters[f_idx, c_idx]
                im = ax.imshow(kernel, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
                ax.set_xticks([])
                ax.set_yticks([])
                if c_idx == 0:
                    ax.set_ylabel(f'F{f_idx}', fontsize=6, family='monospace')
                if f_idx == 0:
                    ax.set_title(f'Ch{c_idx}', fontsize=6, family='monospace')
        
        fig.subplots_adjust(right=0.92, top=0.90, hspace=0.1, wspace=0.1)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Weight')
    else:
        # Single channel filters - show all in a grid
        fig_width = cols * 1.0
        fig_height = rows * 1.0 + 0.8
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98, family='monospace')
        
        vabs = max(abs(filters.min()), abs(filters.max()))
        
        axes_flat = axes.flatten() if num_filters > 1 else [axes]
        
        for idx, ax in enumerate(axes_flat):
            if idx < num_filters:
                kernel = filters[idx, 0]  # Single input channel
                im = ax.imshow(kernel, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(str(idx), fontsize=6, pad=2, family='monospace')
            else:
                ax.axis('off')
        
        fig.subplots_adjust(right=0.92, top=0.90, hspace=0.3, wspace=0.1)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Weight')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_filter_statistics(filters, title, output_path):
    """Create statistics visualization for convolutional filters."""
    num_filters = filters.shape[0]
    
    # Compute statistics per filter
    filter_norms = np.linalg.norm(filters.reshape(num_filters, -1), axis=1)
    filter_means = filters.reshape(num_filters, -1).mean(axis=1)
    filter_stds = filters.reshape(num_filters, -1).std(axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=12, fontweight='bold', family='monospace')
    
    # L2 Norm distribution
    sorted_idx = np.argsort(filter_norms)[::-1]
    axes[0].bar(range(num_filters), filter_norms[sorted_idx], color='#333', width=0.8)
    axes[0].set_xlabel('Filter (sorted)', family='monospace')
    axes[0].set_ylabel('L2 Norm', family='monospace')
    axes[0].set_title('Filter Strength Ranking', family='monospace')
    
    # Mean distribution
    axes[1].hist(filter_means, bins=20, color='#666', edgecolor='#333')
    axes[1].axvline(0, color='#000', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Mean Weight', family='monospace')
    axes[1].set_ylabel('Count', family='monospace')
    axes[1].set_title('Mean Distribution', family='monospace')
    
    # Std distribution
    axes[2].hist(filter_stds, bins=20, color='#666', edgecolor='#333')
    axes[2].set_xlabel('Std Dev', family='monospace')
    axes[2].set_ylabel('Count', family='monospace')
    axes[2].set_title('Std Distribution', family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_conv_filter_visualizations(weights, output_dir):
    """Generate visualizations for convolutional filters."""
    print("\n[1/5] Generating convolutional filter visualizations...")
    
    conv1 = weights['conv1']  # (32, 1, 3, 3)
    conv2 = weights['conv2']  # (64, 32, 3, 3)
    
    print(f"  Conv1 shape: {conv1.shape}")
    print(f"  Conv2 shape: {conv2.shape}")
    
    create_conv_filter_grid(conv1, "Conv Layer 1: 32 Filters (3×3, 1 channel)",
                            os.path.join(output_dir, "01_conv1_filters.png"),
                            grid_shape=(4, 8))
    
    create_conv_filter_grid(conv2, "Conv Layer 2: 64 Filters (3×3, 32 channels)",
                            os.path.join(output_dir, "02_conv2_filters.png"))
    
    create_filter_statistics(conv1, "Conv1 Filter Statistics",
                             os.path.join(output_dir, "03_conv1_stats.png"))
    
    create_filter_statistics(conv2, "Conv2 Filter Statistics",
                             os.path.join(output_dir, "04_conv2_stats.png"))


# =============================================================================
# 2. FILTER PATTERN ANALYSIS
# =============================================================================

def classify_filter_type(kernel):
    """Classify a 3x3 kernel by its pattern type."""
    # Normalize kernel
    k = kernel - kernel.mean()
    if k.std() < 1e-6:
        return "flat"
    k = k / k.std()
    
    # Define pattern templates
    patterns = {
        'horizontal_edge': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        'vertical_edge': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        'diagonal_1': np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
        'diagonal_2': np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]),
        'center': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'blur': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    }
    
    best_match = 'other'
    best_corr = 0.3  # threshold
    
    for name, template in patterns.items():
        t = template - template.mean()
        if t.std() > 1e-6:
            t = t / t.std()
            corr = abs(np.sum(k * t) / 9)
            if corr > best_corr:
                best_corr = corr
                best_match = name
    
    return best_match


def create_filter_type_analysis(filters, title, output_path):
    """Analyze and categorize filter types."""
    num_filters = filters.shape[0]
    
    # For multi-channel, analyze the sum across input channels
    if filters.shape[1] > 1:
        combined = filters.sum(axis=1)
    else:
        combined = filters[:, 0]
    
    # Classify each filter
    types = [classify_filter_type(combined[i]) for i in range(num_filters)]
    type_counts = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=12, fontweight='bold', family='monospace')
    
    # Pie chart of filter types
    labels = list(type_counts.keys())
    sizes = list(type_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    ax1.pie(sizes, labels=labels, autopct='%1.0f%%', colors=colors)
    ax1.set_title('Filter Type Distribution', family='monospace')
    
    # Show example filters for each type
    type_examples = {}
    for i, t in enumerate(types):
        if t not in type_examples:
            type_examples[t] = i
    
    num_types = len(type_examples)
    if num_types > 0:
        for idx, (ftype, fidx) in enumerate(type_examples.items()):
            ax = fig.add_axes([0.55 + (idx % 3) * 0.15, 
                              0.55 - (idx // 3) * 0.4, 0.12, 0.25])
            kernel = combined[fidx]
            vabs = max(abs(kernel.min()), abs(kernel.max()))
            ax.imshow(kernel, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
            ax.set_title(ftype, fontsize=7, family='monospace')
            ax.set_xticks([])
            ax.set_yticks([])
    
    ax2.axis('off')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_filter_analysis(weights, output_dir):
    """Generate filter pattern analysis."""
    print("\n[2/5] Generating filter pattern analysis...")
    
    conv1 = weights['conv1']
    conv2 = weights['conv2']
    
    create_filter_type_analysis(conv1, "Conv1 Filter Type Analysis",
                                 os.path.join(output_dir, "05_conv1_types.png"))
    create_filter_type_analysis(conv2, "Conv2 Filter Type Analysis",
                                 os.path.join(output_dir, "06_conv2_types.png"))


# =============================================================================
# 3. FULLY CONNECTED LAYER ANALYSIS
# =============================================================================

def create_fc_weight_heatmap(W, title, xlabel, ylabel, output_path):
    """Visualize FC weight matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    vabs = max(abs(W.min()), abs(W.max()))
    im = ax.imshow(W, cmap='RdBu_r', vmin=-vabs, vmax=vabs, aspect='auto')
    
    ax.set_title(title, fontsize=12, fontweight='bold', family='monospace')
    ax.set_xlabel(xlabel, fontsize=10, family='monospace')
    ax.set_ylabel(ylabel, fontsize=10, family='monospace')
    
    plt.colorbar(im, ax=ax, label='Weight', shrink=0.8)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_fc_neuron_importance(W, title, output_path):
    """Show which neurons have the strongest outgoing connections."""
    importance = np.linalg.norm(W, axis=0)
    sorted_idx = np.argsort(importance)[::-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=12, fontweight='bold', family='monospace')
    
    ax1.bar(range(len(importance)), importance[sorted_idx], color='#333', width=0.8)
    ax1.set_xlabel('Neuron (sorted)', family='monospace')
    ax1.set_ylabel('L2 Norm', family='monospace')
    ax1.set_title('Importance Ranking', family='monospace')
    
    ax2.hist(importance, bins=25, color='#666', edgecolor='#333')
    ax2.axvline(importance.mean(), color='#000', linestyle='--', linewidth=2,
                label=f'Mean: {importance.mean():.2f}')
    ax2.set_xlabel('L2 Norm', family='monospace')
    ax2.set_ylabel('Count', family='monospace')
    ax2.set_title('Distribution', family='monospace')
    ax2.legend(prop={'family': 'monospace'})
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_spatial_importance_map(fc1_weights, output_path):
    """Visualize which spatial locations in the feature maps are most important."""
    # fc1_weights shape: (128, 3136) where 3136 = 64 * 7 * 7
    # Reshape to see spatial structure: (128, 64, 7, 7)
    
    num_neurons = fc1_weights.shape[0]
    # Sum importance across all neurons and channels to get spatial map
    reshaped = fc1_weights.reshape(num_neurons, 64, 7, 7)
    
    # Compute importance per spatial location (sum of absolute weights)
    spatial_importance = np.abs(reshaped).sum(axis=(0, 1))  # (7, 7)
    
    # Also compute per-channel importance
    channel_importance = np.abs(reshaped).sum(axis=(0, 2, 3))  # (64,)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("FC1 Input Feature Analysis", fontsize=12, fontweight='bold', family='monospace')
    
    # Spatial importance heatmap
    im = ax1.imshow(spatial_importance, cmap='hot')
    ax1.set_title('Spatial Location Importance (7×7)', family='monospace')
    ax1.set_xlabel('Column', family='monospace')
    ax1.set_ylabel('Row', family='monospace')
    plt.colorbar(im, ax=ax1, label='Sum |Weight|', shrink=0.8)
    
    # Channel importance bar chart
    sorted_idx = np.argsort(channel_importance)[::-1]
    ax2.bar(range(64), channel_importance[sorted_idx], color='#333', width=0.8)
    ax2.set_xlabel('Channel (sorted)', family='monospace')
    ax2.set_ylabel('Sum |Weight|', family='monospace')
    ax2.set_title('Channel Importance (64 channels)', family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_fc_analysis(weights, output_dir):
    """Generate FC layer analysis visualizations."""
    print("\n[3/5] Generating fully connected layer analysis...")
    
    fc1 = weights['fc1']  # (128, 3136)
    fc2 = weights['fc2']  # (10, 128)
    
    print(f"  FC1 shape: {fc1.shape}")
    print(f"  FC2 shape: {fc2.shape}")
    
    create_fc_weight_heatmap(fc1, "FC1: Conv Features → Hidden (128 neurons)",
                              "Input Feature (0-3135)", "Hidden Neuron (0-127)",
                              os.path.join(output_dir, "07_fc1_weights.png"))
    
    create_fc_weight_heatmap(fc2, "FC2: Hidden → Output (10 classes)",
                              "Hidden Neuron (0-127)", "Output Class (0-9)",
                              os.path.join(output_dir, "08_fc2_weights.png"))
    
    create_fc_neuron_importance(fc2, "Hidden Layer Neuron Importance",
                                 os.path.join(output_dir, "09_fc_importance.png"))
    
    create_spatial_importance_map(fc1, os.path.join(output_dir, "10_spatial_importance.png"))


# =============================================================================
# 4. CLASS TEMPLATE ANALYSIS
# =============================================================================

def create_class_weight_patterns(fc2, output_path):
    """Show the weight patterns for each class in the output layer."""
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Output Class Weight Patterns (FC2 Weights per Class)", 
                 fontsize=12, fontweight='bold', family='monospace')
    
    vabs = max(abs(fc2.min()), abs(fc2.max()))
    
    for i in range(10):
        row, col = i // 5, i % 5
        ax = axes[row, col]
        
        class_weights = fc2[i]  # (128,)
        
        # Reshape to 8x16 for visualization
        pattern = class_weights.reshape(8, 16)
        im = ax.imshow(pattern, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
        ax.set_title(f"'{i}'", fontsize=11, fontweight='bold', family='monospace')
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Weight')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_class_similarity_matrix(fc2, output_path):
    """Create similarity matrix between class weight patterns."""
    norms = np.linalg.norm(fc2, axis=1, keepdims=True)
    normalized = fc2 / norms
    similarity = normalized @ normalized.T
    
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Class Weight Similarity", fontsize=12, fontweight='bold', family='monospace')
    
    im = ax.imshow(similarity, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel("Digit", family='monospace')
    ax.set_ylabel("Digit", family='monospace')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.8)
    
    for i in range(10):
        for j in range(10):
            color = 'white' if abs(similarity[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{similarity[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color=color, family='monospace')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")
    
    return similarity


def create_class_differences(fc2, output_path):
    """Show what distinguishes commonly confused digit pairs."""
    confused_pairs = [(3, 8), (4, 9), (1, 7), (5, 6), (0, 6)]
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle("Distinguishing Confused Digit Pairs (FC2 Weight Differences)", 
                 fontsize=12, fontweight='bold', family='monospace')
    
    for idx, (d1, d2) in enumerate(confused_pairs):
        weights1 = fc2[d1].reshape(8, 16)
        weights2 = fc2[d2].reshape(8, 16)
        diff = weights1 - weights2
        vabs_diff = max(abs(diff.min()), abs(diff.max()))
        
        ax_diff = axes[0, idx]
        im = ax_diff.imshow(diff, cmap='RdBu_r', vmin=-vabs_diff, vmax=vabs_diff)
        ax_diff.set_title(f"'{d1}' - '{d2}'", fontsize=10, fontweight='bold', family='monospace')
        ax_diff.set_xticks([])
        ax_diff.set_yticks([])
        plt.colorbar(im, ax=ax_diff, shrink=0.7)
        
        ax_both = axes[1, idx]
        combined = np.hstack([weights1, np.ones((8, 1)) * np.nan, weights2])
        vabs = max(abs(fc2.min()), abs(fc2.max()))
        ax_both.imshow(combined, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
        ax_both.set_title(f"'{d1}'    vs    '{d2}'", fontsize=9, family='monospace')
        ax_both.set_xticks([])
        ax_both.set_yticks([])
        ax_both.axvline(x=16, color='black', linewidth=2)
    
    fig.text(0.5, 0.02, "Red = favors first digit | Blue = favors second digit",
             ha='center', fontsize=10, family='monospace', style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_class_analysis(weights, output_dir):
    """Generate class analysis visualizations."""
    print("\n[4/5] Generating class analysis...")
    
    fc2 = weights['fc2']
    
    create_class_weight_patterns(fc2, os.path.join(output_dir, "11_class_patterns.png"))
    similarity = create_class_similarity_matrix(fc2, os.path.join(output_dir, "12_class_similarity.png"))
    create_class_differences(fc2, os.path.join(output_dir, "13_class_differences.png"))
    
    return similarity


# =============================================================================
# 5. STRUCTURED ANALYSIS DATA (JSON) & AI ANALYSIS
# =============================================================================

def compute_analysis_data(weights):
    """Compute all analysis metrics and return as structured data."""
    conv1 = weights['conv1']
    conv2 = weights['conv2']
    fc1 = weights['fc1']
    fc2 = weights['fc2']
    
    def layer_stats(W, name):
        return {
            "name": name,
            "shape": list(W.shape),
            "mean": float(W.mean()),
            "std": float(W.std()),
            "min": float(W.min()),
            "max": float(W.max()),
            "sparsity": float((np.abs(W) < 0.01).mean()),
            "l2_norm": float(np.linalg.norm(W))
        }
    
    # Filter type classification for conv1
    conv1_types = {}
    for i in range(conv1.shape[0]):
        ftype = classify_filter_type(conv1[i, 0])
        conv1_types[ftype] = conv1_types.get(ftype, 0) + 1
    
    # Neuron importance
    fc_importance = np.linalg.norm(fc2, axis=0)
    
    # Class similarity
    norms = np.linalg.norm(fc2, axis=1, keepdims=True)
    normalized = fc2 / norms
    similarity = normalized @ normalized.T
    
    pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            pairs.append({"digits": [i, j], "similarity": float(similarity[i, j])})
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Per-class statistics
    class_stats = []
    for i in range(10):
        w = fc2[i]
        class_stats.append({
            "digit": i,
            "mean": float(w.mean()),
            "std": float(w.std()),
            "min": float(w.min()),
            "max": float(w.max()),
            "l2_norm": float(np.linalg.norm(w)),
            "positive_ratio": float((w > 0).mean())
        })
    
    # Spatial importance from fc1
    reshaped = fc1.reshape(128, 64, 7, 7)
    spatial_importance = np.abs(reshaped).sum(axis=(0, 1))
    center_importance = float(spatial_importance[2:5, 2:5].sum())
    edge_importance = float(spatial_importance.sum() - center_importance)
    
    total_params = conv1.size + conv2.size + fc1.size + fc2.size
    if 'conv1_bias' in weights:
        total_params += weights['conv1_bias'].size
    if 'conv2_bias' in weights:
        total_params += weights['conv2_bias'].size
    if 'fc1_bias' in weights:
        total_params += weights['fc1_bias'].size
    if 'fc2_bias' in weights:
        total_params += weights['fc2_bias'].size
    
    return {
        "model_architecture": {
            "type": "CNN",
            "layers": [
                "Conv2d(1→32, 3×3) + ReLU + MaxPool",
                "Conv2d(32→64, 3×3) + ReLU + MaxPool", 
                "Flatten",
                "Linear(3136→128) + ReLU",
                "Linear(128→10)"
            ],
            "total_parameters": int(total_params),
            "parameter_breakdown": {
                "conv1": int(conv1.size),
                "conv2": int(conv2.size),
                "fc1": int(fc1.size),
                "fc2": int(fc2.size)
            }
        },
        "layer_statistics": {
            "conv1": layer_stats(conv1, "Conv1"),
            "conv2": layer_stats(conv2, "Conv2"),
            "fc1": layer_stats(fc1, "FC1"),
            "fc2": layer_stats(fc2, "FC2")
        },
        "conv_analysis": {
            "conv1_filter_types": conv1_types,
            "conv1_num_filters": int(conv1.shape[0]),
            "conv2_num_filters": int(conv2.shape[0])
        },
        "fc_analysis": {
            "hidden_importance_mean": float(fc_importance.mean()),
            "hidden_importance_std": float(fc_importance.std()),
            "spatial_center_importance": center_importance,
            "spatial_edge_importance": edge_importance,
            "center_vs_edge_ratio": center_importance / edge_importance if edge_importance > 0 else 0
        },
        "class_analysis": {
            "per_class": class_stats,
            "most_similar_pairs": pairs[:5],
            "most_different_pairs": pairs[-5:][::-1],
            "similarity_matrix": similarity.tolist()
        }
    }


def generate_ai_analysis(data):
    """Generate human-readable analysis text based on computed data."""
    analysis = []
    
    # Architecture summary
    total_params = data["model_architecture"]["total_parameters"]
    analysis.append({
        "title": "Model Overview",
        "content": f"This CNN has {total_params:,} trainable parameters. "
                   f"It uses two convolutional layers (32 and 64 filters) with max pooling, "
                   f"followed by fully connected layers. The conv layers extract spatial features "
                   f"while pooling reduces dimensionality from 28×28 to 7×7."
    })
    
    # Conv filter analysis
    conv1_types = data["conv_analysis"]["conv1_filter_types"]
    dominant_type = max(conv1_types.items(), key=lambda x: x[1])[0]
    analysis.append({
        "title": "Convolutional Filters",
        "content": f"Conv1 has {data['conv_analysis']['conv1_num_filters']} filters learning edge and texture patterns. "
                   f"The dominant filter type is '{dominant_type}'. "
                   f"Conv2's {data['conv_analysis']['conv2_num_filters']} filters combine these into higher-level features. "
                   f"The hierarchical structure allows detecting complex digit shapes from simple edges."
    })
    
    # Weight health
    conv1_stats = data["layer_statistics"]["conv1"]
    fc1_stats = data["layer_statistics"]["fc1"]
    
    health = "healthy" if abs(conv1_stats["mean"]) < 0.1 and conv1_stats["std"] < 1.0 else "concerning"
    analysis.append({
        "title": "Weight Distribution",
        "content": f"Layer weights appear {health}. "
                   f"Conv1: mean={conv1_stats['mean']:.4f}, std={conv1_stats['std']:.3f}. "
                   f"FC1: mean={fc1_stats['mean']:.4f}, std={fc1_stats['std']:.3f}. "
                   f"Near-zero means indicate no systematic bias; CNNs typically have tighter weight distributions than MLPs."
    })
    
    # Spatial importance
    fc_analysis = data["fc_analysis"]
    center_ratio = fc_analysis["center_vs_edge_ratio"]
    if center_ratio > 1.5:
        spatial_insight = "The network strongly focuses on center regions where digits are typically centered."
    elif center_ratio > 1.0:
        spatial_insight = "The network slightly favors center regions, consistent with centered digit positioning."
    else:
        spatial_insight = "The network uses edge information heavily, possibly for digit boundary detection."
    
    analysis.append({
        "title": "Spatial Focus",
        "content": f"{spatial_insight} "
                   f"Center importance: {fc_analysis['spatial_center_importance']:.0f}, "
                   f"Edge importance: {fc_analysis['spatial_edge_importance']:.0f} "
                   f"(ratio: {center_ratio:.2f})."
    })
    
    # Class similarity
    most_similar = data["class_analysis"]["most_similar_pairs"][0]
    most_different = data["class_analysis"]["most_different_pairs"][0]
    
    analysis.append({
        "title": "Digit Similarity",
        "content": f"The network sees digits {most_similar['digits'][0]} and {most_similar['digits'][1]} "
                   f"as most similar (cosine={most_similar['similarity']:.3f}), which may cause confusion. "
                   f"Digits {most_different['digits'][0]} and {most_different['digits'][1]} are most distinct "
                   f"(cosine={most_different['similarity']:.3f})."
    })
    
    # CNN vs MLP observation
    analysis.append({
        "title": "CNN Advantage",
        "content": f"Unlike MLPs, CNNs preserve spatial structure through convolutional operations. "
                   f"The 3×3 filters detect local patterns regardless of position (translation invariance). "
                   f"Max pooling provides robustness to small position variations. "
                   f"This architecture is particularly effective for image recognition tasks."
    })
    
    return analysis


def generate_analysis_json(weights, output_dir):
    """Generate structured analysis data as JSON."""
    print("\n[5/5] Generating analysis data...")
    
    data = compute_analysis_data(weights)
    data["ai_analysis"] = generate_ai_analysis(data)
    
    json_path = os.path.join(output_dir, "analysis.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved: {json_path}")
    return data


# =============================================================================
# 6. HTML DASHBOARD
# =============================================================================

def generate_html_dashboard(output_dir, analysis_data):
    """Generate minimalist HTML dashboard with AI analysis."""
    print("\n[6/6] Generating HTML dashboard...")
    
    # Build analysis sections HTML
    analysis_html = ""
    for item in analysis_data["ai_analysis"]:
        analysis_html += f'''
        <div class="analysis-item">
            <h3>{item["title"]}</h3>
            <p>{item["content"]}</p>
        </div>
        '''
    
    # Build similarity pairs
    similar_pairs = analysis_data["class_analysis"]["most_similar_pairs"][:3]
    different_pairs = analysis_data["class_analysis"]["most_different_pairs"][:3]
    
    similar_html = "\n".join([
        f'<div class="pair-row"><span>{p["digits"][0]} ↔ {p["digits"][1]}</span><span>{p["similarity"]:.3f}</span></div>'
        for p in similar_pairs
    ])
    different_html = "\n".join([
        f'<div class="pair-row"><span>{p["digits"][0]} ↔ {p["digits"][1]}</span><span>{p["similarity"]:.3f}</span></div>'
        for p in different_pairs
    ])
    
    # Build class stats table
    class_stats_html = ""
    for c in analysis_data["class_analysis"]["per_class"]:
        class_stats_html += f'''
        <div class="stat-row">
            <span class="digit">{c["digit"]}</span>
            <span>μ={c["mean"]:+.3f}</span>
            <span>σ={c["std"]:.3f}</span>
            <span>‖w‖={c["l2_norm"]:.2f}</span>
        </div>
        '''
    
    # Conv filter type distribution
    filter_types = analysis_data["conv_analysis"]["conv1_filter_types"]
    filter_types_html = "\n".join([
        f'<div class="pair-row"><span>{ftype}</span><span>{count}</span></div>'
        for ftype, count in sorted(filter_types.items(), key=lambda x: -x[1])
    ])
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CNN Model Weight Analysis</title>
    <style>
        body {{
            font-family: monospace;
            max-width: 900px;
            margin: 2rem auto;
            padding: 1rem;
            line-height: 1.6;
        }}
        
        h1 {{
            border-bottom: 2px solid #000;
            padding-bottom: 0.5rem;
        }}
        
        h2 {{
            margin-top: 2.5rem;
            border-bottom: 1px solid #ccc;
            padding-bottom: 0.3rem;
        }}
        
        h3 {{
            margin-bottom: 0.5rem;
            color: #333;
        }}
        
        .section {{
            margin: 1.5rem 0;
        }}
        
        .architecture {{
            padding: 1rem;
            border: 1px solid #ccc;
            margin: 1rem 0;
        }}
        
        .arch-diagram {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.3rem;
            margin: 1rem 0;
            flex-wrap: wrap;
        }}
        
        .layer-box {{
            border: 1px solid #000;
            padding: 0.4rem 0.6rem;
            text-align: center;
            font-size: 0.8rem;
        }}
        
        .layer-box.conv {{
            background: #e3f2fd;
        }}
        
        .layer-box.fc {{
            background: #fff3e0;
        }}
        
        .layer-box .size {{
            font-weight: bold;
        }}
        
        .layer-box .name {{
            font-size: 0.65rem;
            color: #666;
        }}
        
        .arrow {{
            font-size: 1.2rem;
        }}
        
        .params {{
            text-align: center;
            color: #666;
            font-size: 0.875rem;
        }}
        
        .viz-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
            margin: 1rem 0;
        }}
        
        .viz-item {{
            border: 1px solid #ccc;
            padding: 1rem;
        }}
        
        .viz-item h3 {{
            margin-top: 0;
            font-size: 0.875rem;
        }}
        
        .viz-item img {{
            width: 100%;
            cursor: pointer;
        }}
        
        .viz-item img:hover {{
            opacity: 0.9;
        }}
        
        .analysis-item {{
            padding: 1rem;
            border: 1px solid #ccc;
            margin: 1rem 0;
        }}
        
        .analysis-item h3 {{
            margin-top: 0;
        }}
        
        .analysis-item p {{
            margin-bottom: 0;
            color: #333;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .stats-box {{
            border: 1px solid #ccc;
            padding: 1rem;
        }}
        
        .stats-box h4 {{
            margin: 0 0 0.5rem 0;
            font-size: 0.875rem;
        }}
        
        .pair-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            border-bottom: 1px solid #eee;
        }}
        
        .stat-row {{
            display: flex;
            gap: 1rem;
            padding: 0.25rem 0;
            border-bottom: 1px solid #eee;
            font-size: 0.875rem;
        }}
        
        .stat-row .digit {{
            font-weight: bold;
            width: 1rem;
        }}
        
        .note {{
            font-size: 0.75rem;
            color: #666;
            margin-top: 0.5rem;
        }}
        
        footer {{
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #ccc;
            color: #666;
            font-size: 0.875rem;
        }}
        
        /* Lightbox */
        .lightbox {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            cursor: zoom-out;
        }}
        
        .lightbox.active {{
            display: flex;
        }}
        
        .lightbox img {{
            max-width: 95%;
            max-height: 95%;
        }}
    </style>
</head>
<body>
    <h1>CNN Model Weight Analysis</h1>
    <p>Visual and quantitative analysis of the CNN digit recognition network.</p>
    
    <div class="architecture">
        <div class="arch-diagram">
            <div class="layer-box">
                <div class="size">1×28×28</div>
                <div class="name">Input</div>
            </div>
            <div class="arrow">→</div>
            <div class="layer-box conv">
                <div class="size">32×14×14</div>
                <div class="name">Conv1+Pool</div>
            </div>
            <div class="arrow">→</div>
            <div class="layer-box conv">
                <div class="size">64×7×7</div>
                <div class="name">Conv2+Pool</div>
            </div>
            <div class="arrow">→</div>
            <div class="layer-box fc">
                <div class="size">128</div>
                <div class="name">FC1</div>
            </div>
            <div class="arrow">→</div>
            <div class="layer-box fc">
                <div class="size">10</div>
                <div class="name">Output</div>
            </div>
        </div>
        <div class="params">{analysis_data["model_architecture"]["total_parameters"]:,} parameters</div>
    </div>
    
    <h2>AI Analysis</h2>
    <p class="note">Automated insights based on weight statistics and structure.</p>
    {analysis_html}
    
    <h2>Filter Type Distribution</h2>
    <div class="stats-grid">
        <div class="stats-box">
            <h4>Conv1 Filter Types</h4>
            {filter_types_html}
        </div>
        <div class="stats-box">
            <h4>Class Similarity</h4>
            <h5 style="margin: 0.5rem 0 0.25rem 0; font-size: 0.75rem;">Most Similar</h5>
            {similar_html}
            <h5 style="margin: 0.5rem 0 0.25rem 0; font-size: 0.75rem;">Most Different</h5>
            {different_html}
        </div>
    </div>
    
    <h2>Per-Class Statistics</h2>
    <div class="stats-box">
        {class_stats_html}
    </div>
    <p class="note">μ = mean weight, σ = std deviation, ‖w‖ = L2 norm</p>
    
    <h2>Visualizations</h2>
    
    <div class="section">
        <h3>Convolutional Filters</h3>
        <div class="viz-grid">
            <div class="viz-item">
                <h3>Conv1: 32 Filters (3×3)</h3>
                <img src="01_conv1_filters.png" alt="Conv1 Filters" onclick="openLightbox(this)">
                <p class="note">Each filter detects a specific local pattern (edges, textures).</p>
            </div>
            <div class="viz-item">
                <h3>Conv2: 64 Filters (3×3, 32 channels)</h3>
                <img src="02_conv2_filters.png" alt="Conv2 Filters" onclick="openLightbox(this)">
                <p class="note">Higher-level filters combine Conv1 features.</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h3>Filter Analysis</h3>
        <div class="viz-grid">
            <div class="viz-item">
                <h3>Conv1 Statistics</h3>
                <img src="03_conv1_stats.png" alt="Conv1 Stats" onclick="openLightbox(this)">
            </div>
            <div class="viz-item">
                <h3>Conv1 Filter Types</h3>
                <img src="05_conv1_types.png" alt="Conv1 Types" onclick="openLightbox(this)">
            </div>
        </div>
    </div>
    
    <div class="section">
        <h3>Fully Connected Layers</h3>
        <div class="viz-grid">
            <div class="viz-item">
                <h3>FC1 Weight Matrix</h3>
                <img src="07_fc1_weights.png" alt="FC1 Weights" onclick="openLightbox(this)">
            </div>
            <div class="viz-item">
                <h3>FC2 Weight Matrix (Output Layer)</h3>
                <img src="08_fc2_weights.png" alt="FC2 Weights" onclick="openLightbox(this)">
            </div>
            <div class="viz-item">
                <h3>Spatial Importance</h3>
                <img src="10_spatial_importance.png" alt="Spatial Importance" onclick="openLightbox(this)">
                <p class="note">Which spatial locations in the feature maps matter most.</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h3>Class Analysis</h3>
        <div class="viz-grid">
            <div class="viz-item">
                <h3>Class Weight Patterns</h3>
                <img src="11_class_patterns.png" alt="Class Patterns" onclick="openLightbox(this)">
            </div>
            <div class="viz-item">
                <h3>Class Similarity Matrix</h3>
                <img src="12_class_similarity.png" alt="Class Similarity" onclick="openLightbox(this)">
            </div>
            <div class="viz-item">
                <h3>Confused Pair Differences</h3>
                <img src="13_class_differences.png" alt="Class Differences" onclick="openLightbox(this)">
            </div>
        </div>
    </div>
    
    <footer>
        <p>Generated by visualize_cnn.py</p>
        <p>Data exported to <a href="analysis.json">analysis.json</a> for programmatic access.</p>
    </footer>
    
    <div class="lightbox" id="lightbox" onclick="closeLightbox()">
        <img id="lightbox-img" src="" alt="Enlarged">
    </div>
    
    <script>
        function openLightbox(img) {{
            document.getElementById('lightbox-img').src = img.src;
            document.getElementById('lightbox').classList.add('active');
        }}
        
        function closeLightbox() {{
            document.getElementById('lightbox').classList.remove('active');
        }}
        
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeLightbox();
        }});
    </script>
</body>
</html>'''
    
    html_path = os.path.join(output_dir, "dashboard.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"  Saved: {html_path}")
    print(f"\n  → Open in browser: file://{os.path.abspath(html_path)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive CNN weight visualization")
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH,
                        help=f"Path to model weights (default: {WEIGHTS_PATH})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Directory to save visualizations (default: {OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading CNN model from {args.weights}...")
    model = load_trained_model(args.weights)
    weights = get_layer_weights(model)
    
    print(f"Extracted weights:")
    for name, w in weights.items():
        if not name.endswith('_bias'):
            print(f"  {name}: {w.shape}")
    
    generate_conv_filter_visualizations(weights, args.output_dir)
    generate_filter_analysis(weights, args.output_dir)
    generate_fc_analysis(weights, args.output_dir)
    generate_class_analysis(weights, args.output_dir)
    analysis_data = generate_analysis_json(weights, args.output_dir)
    generate_html_dashboard(args.output_dir, analysis_data)
    
    print("\n" + "="*50)
    print("CNN Visualization complete!")
    print("="*50)


if __name__ == "__main__":
    main()

