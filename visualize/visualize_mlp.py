"""
Comprehensive Weight Visualization for Digit Recognition Model.

A multi-perspective visualization strategy that reveals how the network
processes digits from input to output:

1. Layer Overview Grids - All neurons at a glance per layer
2. Connectivity Analysis - How neurons connect across layers  
3. Class Template Analysis - What each digit "looks like" to the network
4. Structured Analysis Data - JSON output for programmatic analysis
5. HTML Dashboard with AI Insights - Human-readable analysis
"""

import argparse
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from common import create_mlp_model

WEIGHTS_PATH = "weights/mlp_weights.pth"
OUTPUT_DIR = "visualizations/mlp"


def load_trained_model(weights_path: str):
    """Load the model with trained weights."""
    model = create_mlp_model()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    return model


def get_weight_matrices(model):
    """Extract weight matrices from each linear layer."""
    weights = []
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            weights.append(layer.weight.detach().numpy())
    return weights


# =============================================================================
# 1. LAYER OVERVIEW GRIDS
# =============================================================================

def create_layer_grid(weights, title, output_path, grid_shape=None, neuron_labels=None):
    """Create a grid showing all neurons in a layer at once."""
    num_neurons = weights.shape[0]
    
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(num_neurons)))
        rows = int(np.ceil(num_neurons / cols))
    else:
        rows, cols = grid_shape
    
    fig_width = cols * 1.2
    fig_height = rows * 1.2 + 0.8
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98, family='monospace')
    
    vabs = max(abs(weights.min()), abs(weights.max()))
    
    axes_flat = axes.flatten() if num_neurons > 1 else [axes]
    
    for idx, ax in enumerate(axes_flat):
        if idx < num_neurons:
            pattern = weights[idx].reshape(28, 28)
            im = ax.imshow(pattern, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
            ax.set_xticks([])
            ax.set_yticks([])
            label = neuron_labels[idx] if neuron_labels else str(idx)
            ax.set_title(label, fontsize=6, pad=2, family='monospace')
        else:
            ax.axis('off')
    
    fig.subplots_adjust(right=0.92, top=0.92, hspace=0.3, wspace=0.1)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Weight')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_layer_overviews(W1, W2, W3, output_dir):
    """Generate overview grids for all layers."""
    print("\n[1/5] Generating layer overview grids...")
    
    create_layer_grid(W1, "Layer 1: Input Feature Detectors (128 neurons)",
                      os.path.join(output_dir, "01_layer1_overview.png"), grid_shape=(8, 16))
    
    W2_composed = W2 @ W1
    create_layer_grid(W2_composed, "Layer 2: Composed Features (W2 × W1)",
                      os.path.join(output_dir, "02_layer2_overview.png"), grid_shape=(8, 16))
    
    W3_composed = W3 @ W2 @ W1
    create_layer_grid(W3_composed, "Layer 3: Digit Class Templates (W3 × W2 × W1)",
                      os.path.join(output_dir, "03_layer3_overview.png"), grid_shape=(2, 5),
                      neuron_labels=[f"'{i}'" for i in range(10)])


# =============================================================================
# 2. CONNECTIVITY ANALYSIS
# =============================================================================

def create_connectivity_heatmap(W, title, xlabel, ylabel, output_path):
    """Visualize weight matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    vabs = max(abs(W.min()), abs(W.max()))
    im = ax.imshow(W, cmap='RdBu_r', vmin=-vabs, vmax=vabs, aspect='auto')
    
    ax.set_title(title, fontsize=12, fontweight='bold', family='monospace')
    ax.set_xlabel(xlabel, fontsize=10, family='monospace')
    ax.set_ylabel(ylabel, fontsize=10, family='monospace')
    
    plt.colorbar(im, ax=ax, label='Weight', shrink=0.8)
    
    ax.set_xticks(np.arange(-0.5, W.shape[1], 10), minor=True)
    ax.set_yticks(np.arange(-0.5, W.shape[0], 10), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_neuron_importance_chart(W, title, output_path):
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


def generate_connectivity_analysis(W1, W2, W3, output_dir):
    """Generate connectivity visualizations."""
    print("\n[2/5] Generating connectivity analysis...")
    
    create_connectivity_heatmap(W1, "W1: Input → Layer 1", "Input Pixel (0-783)",
                                "L1 Neuron (0-127)", os.path.join(output_dir, "04_connectivity_W1.png"))
    create_connectivity_heatmap(W2, "W2: Layer 1 → Layer 2", "L1 Neuron (0-127)",
                                "L2 Neuron (0-127)", os.path.join(output_dir, "05_connectivity_W2.png"))
    create_connectivity_heatmap(W3, "W3: Layer 2 → Output", "L2 Neuron (0-127)",
                                "Output Class (0-9)", os.path.join(output_dir, "06_connectivity_W3.png"))
    
    create_neuron_importance_chart(W2, "Layer 1 Neuron Importance",
                                   os.path.join(output_dir, "07_layer1_importance.png"))
    create_neuron_importance_chart(W3, "Layer 2 Neuron Importance",
                                   os.path.join(output_dir, "08_layer2_importance.png"))


# =============================================================================
# 3. CLASS TEMPLATE ANALYSIS
# =============================================================================

def create_class_comparison(W3_composed, output_path):
    """Create comparison of all 10 digit class templates."""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 5, hspace=0.3, wspace=0.2)
    
    fig.suptitle("Digit Class Templates", fontsize=12, fontweight='bold', 
                 y=0.98, family='monospace')
    
    vabs = max(abs(W3_composed.min()), abs(W3_composed.max()))
    
    for i in range(10):
        row, col = i // 5, i % 5
        ax = fig.add_subplot(gs[row, col])
        pattern = W3_composed[i].reshape(28, 28)
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


def create_class_differences(W3_composed, output_path):
    """Show what distinguishes commonly confused digit pairs."""
    confused_pairs = [(3, 8), (4, 9), (1, 7), (5, 6), (0, 6)]
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle("Distinguishing Confused Digit Pairs", fontsize=12, 
                 fontweight='bold', family='monospace')
    
    for idx, (d1, d2) in enumerate(confused_pairs):
        pattern1 = W3_composed[d1].reshape(28, 28)
        pattern2 = W3_composed[d2].reshape(28, 28)
        diff = pattern1 - pattern2
        vabs_diff = max(abs(diff.min()), abs(diff.max()))
        
        ax_diff = axes[0, idx]
        im = ax_diff.imshow(diff, cmap='RdBu_r', vmin=-vabs_diff, vmax=vabs_diff)
        ax_diff.set_title(f"'{d1}' - '{d2}'", fontsize=10, fontweight='bold', family='monospace')
        ax_diff.set_xticks([])
        ax_diff.set_yticks([])
        plt.colorbar(im, ax=ax_diff, shrink=0.7)
        
        ax_both = axes[1, idx]
        combined = np.hstack([pattern1, np.ones((28, 2)) * np.nan, pattern2])
        vabs = max(abs(W3_composed.min()), abs(W3_composed.max()))
        ax_both.imshow(combined, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
        ax_both.set_title(f"'{d1}'    vs    '{d2}'", fontsize=9, family='monospace')
        ax_both.set_xticks([])
        ax_both.set_yticks([])
        ax_both.axvline(x=28.5, color='black', linewidth=2)
    
    fig.text(0.5, 0.02, "Red = favors first digit | Blue = favors second digit",
             ha='center', fontsize=10, family='monospace', style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def create_class_similarity_matrix(W3_composed, output_path):
    """Create similarity matrix between class templates."""
    norms = np.linalg.norm(W3_composed, axis=1, keepdims=True)
    normalized = W3_composed / norms
    similarity = normalized @ normalized.T
    
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Class Template Similarity", fontsize=12, fontweight='bold', family='monospace')
    
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


def generate_class_analysis(W1, W2, W3, output_dir):
    """Generate class template analysis visualizations."""
    print("\n[3/5] Generating class template analysis...")
    
    W3_composed = W3 @ W2 @ W1
    
    create_class_comparison(W3_composed, os.path.join(output_dir, "09_class_templates.png"))
    create_class_differences(W3_composed, os.path.join(output_dir, "10_class_differences.png"))
    similarity = create_class_similarity_matrix(W3_composed, os.path.join(output_dir, "11_class_similarity.png"))
    
    return similarity


# =============================================================================
# 4. STRUCTURED ANALYSIS DATA (JSON)
# =============================================================================

def compute_analysis_data(W1, W2, W3):
    """Compute all analysis metrics and return as structured data."""
    W2_composed = W2 @ W1
    W3_composed = W3 @ W2 @ W1
    
    # Layer statistics
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
    
    # Neuron importance (L2 norm of outgoing weights)
    l1_importance = np.linalg.norm(W2, axis=0)
    l2_importance = np.linalg.norm(W3, axis=0)
    
    # Class similarity
    norms = np.linalg.norm(W3_composed, axis=1, keepdims=True)
    normalized = W3_composed / norms
    similarity = normalized @ normalized.T
    
    # Find most/least similar pairs
    pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            pairs.append({"digits": [i, j], "similarity": float(similarity[i, j])})
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Per-class statistics
    class_stats = []
    for i in range(10):
        w = W3_composed[i]
        class_stats.append({
            "digit": i,
            "mean": float(w.mean()),
            "std": float(w.std()),
            "min": float(w.min()),
            "max": float(w.max()),
            "l2_norm": float(np.linalg.norm(w)),
            "positive_ratio": float((w > 0).mean()),
            "peak_location": [int(x) for x in np.unravel_index(np.argmax(w), (28, 28))]
        })
    
    # Dead neuron analysis
    l1_dead = int((l1_importance < l1_importance.mean() * 0.3).sum())
    l2_dead = int((l2_importance < l2_importance.mean() * 0.3).sum())
    
    return {
        "model_architecture": {
            "layers": ["784 (input)", "128 (hidden1 + ReLU)", "128 (hidden2 + ReLU)", "10 (output)"],
            "total_parameters": int(W1.size + W2.size + W3.size),
            "parameter_breakdown": {
                "W1": int(W1.size),
                "W2": int(W2.size),
                "W3": int(W3.size)
            }
        },
        "layer_statistics": {
            "W1_direct": layer_stats(W1, "W1: Input → Hidden1"),
            "W2_direct": layer_stats(W2, "W2: Hidden1 → Hidden2"),
            "W3_direct": layer_stats(W3, "W3: Hidden2 → Output"),
            "W2_composed": layer_stats(W2_composed, "W2×W1: Composed L2"),
            "W3_composed": layer_stats(W3_composed, "W3×W2×W1: Class Templates")
        },
        "neuron_importance": {
            "layer1": {
                "mean": float(l1_importance.mean()),
                "std": float(l1_importance.std()),
                "min": float(l1_importance.min()),
                "max": float(l1_importance.max()),
                "potentially_dead": l1_dead,
                "top_5_indices": [int(x) for x in np.argsort(l1_importance)[-5:][::-1]]
            },
            "layer2": {
                "mean": float(l2_importance.mean()),
                "std": float(l2_importance.std()),
                "min": float(l2_importance.min()),
                "max": float(l2_importance.max()),
                "potentially_dead": l2_dead,
                "top_5_indices": [int(x) for x in np.argsort(l2_importance)[-5:][::-1]]
            }
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
        "content": f"This is a 3-layer MLP with {total_params:,} trainable parameters. "
                   f"The architecture (784→128→128→10) is compact but sufficient for MNIST, "
                   f"which has relatively simple digit patterns."
    })
    
    # Weight distribution analysis
    w1_stats = data["layer_statistics"]["W1_direct"]
    w2_stats = data["layer_statistics"]["W2_direct"]
    w3_stats = data["layer_statistics"]["W3_direct"]
    
    weight_health = "healthy" if abs(w1_stats["mean"]) < 0.1 and w1_stats["std"] < 0.5 else "concerning"
    analysis.append({
        "title": "Weight Distribution",
        "content": f"Layer weights appear {weight_health}. "
                   f"W1 has mean={w1_stats['mean']:.4f}, std={w1_stats['std']:.3f}. "
                   f"W2 has mean={w2_stats['mean']:.4f}, std={w2_stats['std']:.3f}. "
                   f"W3 has mean={w3_stats['mean']:.4f}, std={w3_stats['std']:.3f}. "
                   f"Near-zero means suggest no systematic bias; moderate std indicates active learning."
    })
    
    # Neuron health
    l1_dead = data["neuron_importance"]["layer1"]["potentially_dead"]
    l2_dead = data["neuron_importance"]["layer2"]["potentially_dead"]
    
    if l1_dead == 0 and l2_dead == 0:
        neuron_status = "All neurons appear active and contributing."
    else:
        neuron_status = f"Found {l1_dead} potentially weak neurons in L1 and {l2_dead} in L2."
    
    top_l1 = data["neuron_importance"]["layer1"]["top_5_indices"]
    analysis.append({
        "title": "Neuron Activity",
        "content": f"{neuron_status} "
                   f"The most influential L1 neurons are {top_l1}, "
                   f"which have the strongest connections to Layer 2. "
                   f"A smooth importance distribution (no extreme outliers) suggests balanced training."
    })
    
    # Class template insights
    most_similar = data["class_analysis"]["most_similar_pairs"][0]
    most_different = data["class_analysis"]["most_different_pairs"][0]
    
    analysis.append({
        "title": "Digit Similarity",
        "content": f"The network sees digits {most_similar['digits'][0]} and {most_similar['digits'][1]} "
                   f"as most similar (cosine={most_similar['similarity']:.3f}), which may lead to confusion. "
                   f"Digits {most_different['digits'][0]} and {most_different['digits'][1]} are most distinct "
                   f"(cosine={most_different['similarity']:.3f}), meaning the network has learned clear "
                   f"distinguishing features for these."
    })
    
    # Class-specific observations
    class_stats = data["class_analysis"]["per_class"]
    highest_norm = max(class_stats, key=lambda x: x["l2_norm"])
    lowest_norm = min(class_stats, key=lambda x: x["l2_norm"])
    
    analysis.append({
        "title": "Class Templates",
        "content": f"Digit '{highest_norm['digit']}' has the strongest template (L2={highest_norm['l2_norm']:.2f}), "
                   f"suggesting the network is most confident about its defining features. "
                   f"Digit '{lowest_norm['digit']}' has the weakest template (L2={lowest_norm['l2_norm']:.2f}), "
                   f"which may indicate it's defined more by the absence of features from other digits. "
                   f"Template peaks (bright spots) indicate where the network expects ink for each digit."
    })
    
    # Recommendations
    sparsity = data["layer_statistics"]["W1_direct"]["sparsity"]
    analysis.append({
        "title": "Observations",
        "content": f"Weight sparsity is {sparsity:.1%} (fraction near zero). "
                   f"The composed templates (W3×W2×W1) show recognizable digit-like patterns, "
                   f"indicating the network has learned meaningful representations. "
                   f"The linear composition is an approximation—actual network behavior includes "
                   f"ReLU nonlinearities that gate which patterns are active for each input."
    })
    
    return analysis


def generate_analysis_json(W1, W2, W3, output_dir):
    """Generate structured analysis data as JSON."""
    print("\n[4/5] Generating analysis data...")
    
    data = compute_analysis_data(W1, W2, W3)
    data["ai_analysis"] = generate_ai_analysis(data)
    
    json_path = os.path.join(output_dir, "analysis.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved: {json_path}")
    return data


# =============================================================================
# 5. HTML DASHBOARD
# =============================================================================

def generate_html_dashboard(output_dir, analysis_data):
    """Generate minimalist HTML dashboard with AI analysis."""
    print("\n[5/5] Generating HTML dashboard...")
    
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
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Weight Analysis</title>
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
            gap: 0.5rem;
            margin: 1rem 0;
            flex-wrap: wrap;
        }}
        
        .layer-box {{
            border: 1px solid #000;
            padding: 0.5rem 1rem;
            text-align: center;
        }}
        
        .layer-box .size {{
            font-size: 1.25rem;
            font-weight: bold;
        }}
        
        .layer-box .name {{
            font-size: 0.75rem;
            color: #666;
        }}
        
        .arrow {{
            font-size: 1.5rem;
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
    <h1>Model Weight Analysis</h1>
    <p>Visual and quantitative analysis of the digit recognition neural network.</p>
    
    <div class="architecture">
        <div class="arch-diagram">
            <div class="layer-box">
                <div class="size">784</div>
                <div class="name">Input (28×28)</div>
            </div>
            <div class="arrow">→</div>
            <div class="layer-box">
                <div class="size">128</div>
                <div class="name">Hidden 1</div>
            </div>
            <div class="arrow">→</div>
            <div class="layer-box">
                <div class="size">128</div>
                <div class="name">Hidden 2</div>
            </div>
            <div class="arrow">→</div>
            <div class="layer-box">
                <div class="size">10</div>
                <div class="name">Output</div>
            </div>
        </div>
        <div class="params">{analysis_data["model_architecture"]["total_parameters"]:,} parameters</div>
    </div>
    
    <h2>AI Analysis</h2>
    <p class="note">Automated insights based on weight statistics and structure.</p>
    {analysis_html}
    
    <h2>Class Similarity</h2>
    <div class="stats-grid">
        <div class="stats-box">
            <h4>Most Similar Pairs</h4>
            {similar_html}
        </div>
        <div class="stats-box">
            <h4>Most Different Pairs</h4>
            {different_html}
        </div>
    </div>
    
    <h2>Per-Class Statistics</h2>
    <div class="stats-box">
        {class_stats_html}
    </div>
    <p class="note">μ = mean weight, σ = std deviation, ‖w‖ = L2 norm (template strength)</p>
    
    <h2>Visualizations</h2>
    
    <div class="section">
        <h3>Layer Overviews</h3>
        <div class="viz-grid">
            <div class="viz-item">
                <h3>Layer 1: Feature Detectors (128 neurons)</h3>
                <img src="01_layer1_overview.png" alt="Layer 1" onclick="openLightbox(this)">
                <p class="note">Direct weights showing what input patterns each neuron detects.</p>
            </div>
            <div class="viz-item">
                <h3>Layer 2: Composed Features (W2×W1)</h3>
                <img src="02_layer2_overview.png" alt="Layer 2" onclick="openLightbox(this)">
                <p class="note">Effective input patterns for layer 2 neurons.</p>
            </div>
            <div class="viz-item">
                <h3>Layer 3: Class Templates (W3×W2×W1)</h3>
                <img src="03_layer3_overview.png" alt="Layer 3" onclick="openLightbox(this)">
                <p class="note">What each digit "looks like" to the network.</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h3>Class Analysis</h3>
        <div class="viz-grid">
            <div class="viz-item">
                <h3>Digit Templates</h3>
                <img src="09_class_templates.png" alt="Templates" onclick="openLightbox(this)">
            </div>
            <div class="viz-item">
                <h3>Confused Pairs Differences</h3>
                <img src="10_class_differences.png" alt="Differences" onclick="openLightbox(this)">
            </div>
            <div class="viz-item">
                <h3>Similarity Matrix</h3>
                <img src="11_class_similarity.png" alt="Similarity" onclick="openLightbox(this)">
            </div>
        </div>
    </div>
    
    <div class="section">
        <h3>Connectivity</h3>
        <div class="viz-grid">
            <div class="viz-item">
                <h3>W2: Layer 1 → Layer 2</h3>
                <img src="05_connectivity_W2.png" alt="W2" onclick="openLightbox(this)">
            </div>
            <div class="viz-item">
                <h3>W3: Layer 2 → Output</h3>
                <img src="06_connectivity_W3.png" alt="W3" onclick="openLightbox(this)">
            </div>
        </div>
    </div>
    
    <div class="section">
        <h3>Neuron Importance</h3>
        <div class="viz-grid">
            <div class="viz-item">
                <h3>Layer 1 Importance</h3>
                <img src="07_layer1_importance.png" alt="L1 Importance" onclick="openLightbox(this)">
            </div>
            <div class="viz-item">
                <h3>Layer 2 Importance</h3>
                <img src="08_layer2_importance.png" alt="L2 Importance" onclick="openLightbox(this)">
            </div>
        </div>
    </div>
    
    <footer>
        <p>Generated by visualize.py</p>
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
    parser = argparse.ArgumentParser(description="Comprehensive neural network weight visualization")
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH,
                        help=f"Path to model weights (default: {WEIGHTS_PATH})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Directory to save visualizations (default: {OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.weights}...")
    model = load_trained_model(args.weights)
    W1, W2, W3 = get_weight_matrices(model)
    print(f"Weight shapes: W1={W1.shape}, W2={W2.shape}, W3={W3.shape}")
    
    generate_layer_overviews(W1, W2, W3, args.output_dir)
    generate_connectivity_analysis(W1, W2, W3, args.output_dir)
    generate_class_analysis(W1, W2, W3, args.output_dir)
    analysis_data = generate_analysis_json(W1, W2, W3, args.output_dir)
    generate_html_dashboard(args.output_dir, analysis_data)
    
    print("\n" + "="*50)
    print("Visualization complete!")
    print("="*50)


if __name__ == "__main__":
    main()
