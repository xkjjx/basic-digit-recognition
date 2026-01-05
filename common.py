import struct
import torch
import torch.nn as nn

def create_mlp_model():
    """Create a simple MLP neural network architecture."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

def create_cnn_model():
    """Create a CNN architecture for MNIST digit recognition."""
    return nn.Sequential(
        # First conv block: 1 -> 32 channels
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
        
        # Second conv block: 32 -> 64 channels
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
        
        # Flatten and fully connected layers
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

def load_mnist_images(filepath):
    """Load MNIST images from IDX file format."""
    with open(filepath, "rb") as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read pixel data
        data = f.read()
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num_images, rows, cols).float() / 255.0
    return images


def load_mnist_labels(filepath):
    """Load MNIST labels from IDX file format."""
    with open(filepath, "rb") as f:
        # Read header
        magic, num_labels = struct.unpack(">II", f.read(8))
        # Read label data
        data = f.read()
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return labels