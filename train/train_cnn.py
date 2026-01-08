import argparse
import json
import torch
import torch.nn as nn
from common import augment_with_rotations, create_cnn_model, load_mnist_images, load_mnist_labels

TRAINING_IMAGES_PATH = "data/train-images.idx3-ubyte"
TRAINING_LABELS_PATH = "data/train-labels.idx1-ubyte"
MODEL_SAVE_PATH = "weights/cnn_weights"


def save_model(model, save_path, format):
    """Save model in the specified format."""
    if format == "pth":
        output_path = f"{save_path}.pth"
        torch.save(model.state_dict(), output_path)
    elif format == "onnx":
        output_path = f"{save_path}.onnx"
        model.eval()
        dummy_input = torch.randn(1, 1, 28, 28)  # (batch, channels, height, width)
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


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_scheduler(scheduler_type, optimizer, epochs):
    """Create a learning rate scheduler based on the specified type."""
    if scheduler_type == "none":
        return None
    elif scheduler_type == "step":
        # Reduce LR by 10x every 30 epochs
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == "cosine":
        # Smoothly decay LR following cosine curve
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "exponential":
        # Multiply LR by 0.95 each epoch
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == "onecycle":
        # 1cycle policy: ramp up then down (good for fast convergence)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=1
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a CNN digit recognition neural network")
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
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "step", "cosine", "exponential", "onecycle"],
        default="cosine",
        help="Learning rate scheduler: none, step, cosine, exponential, onecycle (default: cosine)",
    )
    parser.add_argument(
        "--num-rotations",
        type=int,
        default=0,
        help="Number of rotation angles for data augmentation (default: 0, disabled)",
    )
    args = parser.parse_args()
    
    # Select device (MPS for Apple Silicon GPU acceleration)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create the neural network and move to device
    model = create_cnn_model().to(device)
    
    # Load weights if specified, otherwise use random initialization
    if args.load_weights:
        model.load_state_dict(torch.load(args.load_weights, weights_only=True))
        print(f"Loaded weights from {args.load_weights}")
    else:
        print("Using random weight initialization")
    
    # Load training data and move to device
    images = load_mnist_images(args.training_images).to(device)
    labels = load_mnist_labels(args.training_labels).to(device)

    # Apply rotation augmentation if enabled
    if args.num_rotations > 0:
        original_size = len(images)
        images, labels = augment_with_rotations(images, labels, args.num_rotations)
        print(f"Augmented dataset with {args.num_rotations} rotations: {original_size} -> {len(images)} images")

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = create_scheduler(args.scheduler, optimizer, args.epochs)
    if scheduler:
        print(f"Using {args.scheduler} learning rate scheduler")
    
    # Training loop with backpropagation
    epochs = args.epochs
    batch_size = args.batch_size
    num_batches = len(images) // batch_size
    
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(num_batches):
            # Get batch
            start = i * batch_size
            end = start + batch_size
            batch_images = images[start:end]
            batch_labels = labels[start:end]
            
            # Reshape for CNN: (batch, 28, 28) -> (batch, 1, 28, 28)
            batch_images = batch_images.unsqueeze(1)
            
            # Forward pass
            output = model(batch_images)
            loss = loss_fn(output, batch_labels)
            
            # Backward pass (backpropagation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Step the scheduler after each epoch
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.8f}, LR: {current_lr:.8f}")
    
    # Move model back to CPU for saving (required for ONNX export)
    model = model.to("cpu")
    
    # Save model weights after training
    save_model(model, args.save_path, args.format)


if __name__ == "__main__":
    main()

