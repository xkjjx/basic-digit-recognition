import struct
import torch
import torch.nn as nn

def create_model():
    """Create the neural network architecture."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
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