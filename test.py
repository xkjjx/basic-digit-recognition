import argparse
import torch
from common import create_model, load_mnist_images, load_mnist_labels

TEST_IMAGES_PATH = "data/t10k-images.idx3-ubyte"
TEST_LABELS_PATH = "data/t10k-labels.idx1-ubyte"


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test a digit recognition neural network")
    parser.add_argument(
        "weights_file",
        type=str,
        help="Path to the model weights file",
    )
    parser.add_argument(
        "--test-images",
        type=str,
        default=TEST_IMAGES_PATH,
        help=f"Path to test images file (default: {TEST_IMAGES_PATH})",
    )
    parser.add_argument(
        "--test-labels",
        type=str,
        default=TEST_LABELS_PATH,
        help=f"Path to test labels file (default: {TEST_LABELS_PATH})",
    )
    args = parser.parse_args()
    
    # Create the neural network and load weights
    model = create_model()
    model.load_state_dict(torch.load(args.weights_file, weights_only=True))
    model.eval()  # Set to evaluation mode
    print(f"Loaded weights from {args.weights_file}")
    
    # Load test data
    images = load_mnist_images(args.test_images)
    labels = load_mnist_labels(args.test_labels)
    print(f"Loaded {len(images)} test images")
    
    # Run inference
    with torch.no_grad():
        output = model(images)
        predictions = output.argmax(dim=1)
    
    # Calculate accuracy
    correct = (predictions == labels).sum().item()
    total = len(labels)
    accuracy = correct / total * 100
    
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()

