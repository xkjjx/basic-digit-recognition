import secrets
import struct
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

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


def generate_model_filename(model_type, epochs, lr, batch_size, scheduler, num_rotations):
    """Generate a descriptive filename with model characteristics and random hash."""
    sched_abbrev = {"none": "none", "step": "step", "cosine": "cos", "exponential": "exp", "onecycle": "1cyc"}
    sched = sched_abbrev.get(scheduler, scheduler[:4])
    hash_suffix = secrets.token_hex(3)
    name = f"{model_type}_e{epochs}_lr{lr}_bs{batch_size}_{sched}"
    if num_rotations > 0:
        name += f"_rot{num_rotations}"
    name += f"_{hash_suffix}"
    return f"weights/{name}"


def augment_with_rotations(images, labels, num_rotations):
    """Augment dataset by creating rotated copies of images.

    Args:
        images: Tensor of shape (N, 28, 28)
        labels: Tensor of shape (N,)
        num_rotations: Number of rotated versions to create per image (evenly spaced from -180 to +180 degrees)

    Returns:
        Augmented images and labels tensors
    """
    if num_rotations <= 0:
        return images, labels

    augmented_images = [images]
    augmented_labels = [labels]

    angles = torch.linspace(-180, 180, num_rotations + 1)[:-1]  # Exclude 180 since it equals -180

    for angle in angles:
        rotated = TF.rotate(images.unsqueeze(1), angle.item()).squeeze(1)
        augmented_images.append(rotated)
        augmented_labels.append(labels.clone())

    all_images = torch.cat(augmented_images, dim=0)
    all_labels = torch.cat(augmented_labels, dim=0)

    # Shuffle the augmented dataset
    perm = torch.randperm(len(all_images))
    return all_images[perm], all_labels[perm]