import argparse
import json
import torch
import torch.nn as nn
from common import create_mlp_model, load_mnist_images, load_mnist_labels

TRAINING_IMAGES_PATH = "data/train-images.idx3-ubyte"
TRAINING_LABELS_PATH = "data/train-labels.idx1-ubyte"
MODEL_SAVE_PATH = "weights/mlp_weights"


def save_model(model, save_path, format):
    """Save model in the specified format."""
    if format == "pth":
        output_path = f"{save_path}.pth"
        torch.save(model.state_dict(), output_path)
    elif format == "onnx":
        output_path = f"{save_path}.onnx"
        model.eval()
        dummy_input = torch.randn(1, 28, 28)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
    elif format == "json":
        output_path = f"{save_path}.json"
        state_dict = model.state_dict()
        json_dict = {key: tensor.tolist() for key, tensor in state_dict.items()}
        with open(output_path, "w") as f:
            json.dump(json_dict, f)
    
    print(f"Model saved to {output_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train an MLP digit recognition neural network")
    parser.add_argument(
        "--load-weights",
        type=str,
        help="Load model weights from the specified file instead of random initialization",
    )
    parser.add_argument(
        "--training-images",
        type=str,
        default=TRAINING_IMAGES_PATH,
        help=f"Path to training images file (default: {TRAINING_IMAGES_PATH})",
    )
    parser.add_argument(
        "--training-labels",
        type=str,
        default=TRAINING_LABELS_PATH,
        help=f"Path to training labels file (default: {TRAINING_LABELS_PATH})",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=MODEL_SAVE_PATH,
        help=f"Path to save model weights without extension (default: {MODEL_SAVE_PATH})",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pth", "onnx", "json"],
        default="pth",
        help="Format to save model weights: pth (PyTorch), onnx, or json (default: pth)",
    )
    args = parser.parse_args()
    
    # Create the neural network
    model = create_mlp_model()
    
    # Load weights if specified, otherwise use random initialization
    if args.load_weights:
        model.load_state_dict(torch.load(args.load_weights, weights_only=True))
        print(f"Loaded weights from {args.load_weights}")
    else:
        print("Using random weight initialization")
    
    # Load training data
    images = load_mnist_images(args.training_images)
    labels = load_mnist_labels(args.training_labels)
    
    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Training loop with backpropagation
    epochs = 100
    batch_size = 64
    num_batches = len(images) // batch_size
    
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(num_batches):
            # Get batch
            start = i * batch_size
            end = start + batch_size
            batch_images = images[start:end]
            batch_labels = labels[start:end]
            
            # Forward pass
            output = model(batch_images)
            loss = loss_fn(output, batch_labels)
            
            # Backward pass (backpropagation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Save model weights after training
    save_model(model, args.save_path, args.format)


if __name__ == "__main__":
    main()

