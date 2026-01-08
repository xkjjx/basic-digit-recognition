import argparse
import subprocess
import sys

# Default paths
TRAINING_IMAGES_PATH = "data/train-images.idx3-ubyte"
TRAINING_LABELS_PATH = "data/train-labels.idx1-ubyte"
TEST_IMAGES_PATH = "data/t10k-images.idx3-ubyte"
TEST_LABELS_PATH = "data/t10k-labels.idx1-ubyte"

# Default training hyperparameters
DEFAULT_LR = 0.001
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_SCHEDULER = "cosine"


def main():
    parser = argparse.ArgumentParser(
        description="Train and test a digit recognition neural network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_and_test.py --model cnn
  python train_and_test.py --model mlp --epochs 50 --lr 0.0005
  python train_and_test.py --model cnn --scheduler step --batch-size 128
        """
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "cnn"],
        required=True,
        help="Model architecture to train: mlp or cnn",
    )

    # Training arguments (passed through to training script)
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
        help="Path to save model weights without extension (default: weights/<model>_weights)",
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
        default=DEFAULT_LR,
        help=f"Initial learning rate (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "step", "cosine", "exponential", "onecycle"],
        default=DEFAULT_SCHEDULER,
        help=f"Learning rate scheduler (default: {DEFAULT_SCHEDULER})",
    )

    # Test arguments
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
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip testing after training",
    )

    args = parser.parse_args()

    # Build training command
    train_script = f"train.train_{args.model}"
    train_cmd = [sys.executable, "-m", train_script]

    # Add training arguments
    if args.load_weights:
        train_cmd.extend(["--load-weights", args.load_weights])
    train_cmd.extend(["--training-images", args.training_images])
    train_cmd.extend(["--training-labels", args.training_labels])
    if args.save_path:
        train_cmd.extend(["--save-path", args.save_path])
    train_cmd.extend(["--format", args.format])
    train_cmd.extend(["--lr", str(args.lr)])
    train_cmd.extend(["--epochs", str(args.epochs)])
    train_cmd.extend(["--batch-size", str(args.batch_size)])
    train_cmd.extend(["--scheduler", args.scheduler])

    # Determine weights path for testing
    if args.save_path:
        weights_path = f"{args.save_path}.pth"
    else:
        weights_path = f"weights/{args.model}_weights.pth"

    # Run training
    print(f"=== Training {args.model.upper()} model ===")
    print(f"Running: {' '.join(train_cmd)}\n")

    train_result = subprocess.run(train_cmd)

    if train_result.returncode != 0:
        print(f"\nTraining failed with exit code {train_result.returncode}")
        sys.exit(train_result.returncode)

    # Skip test if requested or if format is not pth
    if args.skip_test:
        print("\nSkipping test (--skip-test flag provided)")
        return

    if args.format != "pth":
        print(f"\nSkipping test (model saved in {args.format} format, testing requires .pth)")
        return

    # Run testing
    print(f"\n=== Testing {args.model.upper()} model ===")

    test_cmd = [sys.executable, "test.py", weights_path, "--model-type", args.model]
    test_cmd.extend(["--test-images", args.test_images])
    test_cmd.extend(["--test-labels", args.test_labels])

    print(f"Running: {' '.join(test_cmd)}\n")

    test_result = subprocess.run(test_cmd)

    if test_result.returncode != 0:
        print(f"\nTesting failed with exit code {test_result.returncode}")
        sys.exit(test_result.returncode)

    print("\n=== Training and testing completed successfully ===")


if __name__ == "__main__":
    main()
