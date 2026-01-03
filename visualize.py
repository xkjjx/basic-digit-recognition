"""
Weight Visualization for Digit Recognition Model.

Visualizes first layer weights as 28x28 images showing what patterns each neuron detects.
"""

import argparse
import os
import torch
import matplotlib.pyplot as plt
from common import create_model

WEIGHTS_PATH = "weights/model_weights.pth"
OUTPUT_DIR = "visualizations"


def load_trained_model(weights_path: str):
    """Load the model with trained weights."""
    model = create_model()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    return model


def visualize_first_layer_weights(model, output_path: str):
    """
    Visualize first layer weights as 28x28 images.
    
    Each of the 128 neurons in the first layer has 784 weights,
    which can be reshaped into 28x28 images showing what visual
    patterns that neuron has learned to detect.
    """
    # Get first layer weights: shape (128, 784)
    first_layer = list(model.children())[1]  # Index 1 because 0 is Flatten
    weights = first_layer.weight.detach().numpy()
    
    # Create a grid of 128 neurons (16x8 grid)
    fig, axes = plt.subplots(8, 16, figsize=(20, 10))
    fig.suptitle("First Layer Weights (128 neurons detecting input patterns)", 
                 fontsize=14, fontweight='bold')
    
    # Normalize weights for visualization
    vmin, vmax = weights.min(), weights.max()
    vabs = max(abs(vmin), abs(vmax))
    
    for idx, ax in enumerate(axes.flat):
        if idx < weights.shape[0]:
            # Reshape 784 weights to 28x28 image
            neuron_weights = weights[idx].reshape(28, 28)
            im = ax.imshow(neuron_weights, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{idx}', fontsize=6)
        else:
            ax.axis('off')
    
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Weight Value')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved first layer weights visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize first layer neural network weights"
    )
    parser.add_argument(
        "--weights", type=str, default=WEIGHTS_PATH,
        help=f"Path to model weights (default: {WEIGHTS_PATH})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Directory to save visualization (default: {OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.weights}...")
    model = load_trained_model(args.weights)
    
    # Generate visualization
    visualize_first_layer_weights(
        model, 
        os.path.join(args.output_dir, "first_layer_weights.png")
    )
    
    print(f"\nVisualization saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
