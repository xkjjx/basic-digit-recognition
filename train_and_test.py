import argparse
import os
import re
import smtplib
import subprocess
import sys
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

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


def send_email(subject, body):
    """Send email notification to SMTP_USER using SMTP settings from environment variables."""
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        print("Warning: SMTP_USER and SMTP_PASSWORD environment variables not set, skipping email")
        return False

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = smtp_user

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, smtp_user, msg.as_string())
        print(f"Email notification sent to {smtp_user}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


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
        help="Path to save model weights without extension (default: auto-generated)",
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
    parser.add_argument(
        "--num-rotations",
        type=int,
        default=0,
        help="Number of rotation angles for data augmentation (default: 0, disabled)",
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
    parser.add_argument(
        "--email",
        action="store_true",
        help="Send email notification when training completes (requires SMTP_USER, SMTP_PASSWORD)",
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
    if args.num_rotations > 0:
        train_cmd.extend(["--num-rotations", str(args.num_rotations)])

    # Run training
    print(f"=== Training {args.model.upper()} model ===")
    print(f"Running: {' '.join(train_cmd)}\n")

    train_result = subprocess.run(train_cmd, capture_output=True, text=True)
    print(train_result.stdout)
    if train_result.stderr:
        print(train_result.stderr, file=sys.stderr)

    if train_result.returncode != 0:
        print(f"\nTraining failed with exit code {train_result.returncode}")
        sys.exit(train_result.returncode)

    # Parse weights path from training output
    match = re.search(r"Model saved to (.+\.pth)", train_result.stdout)
    weights_path = match.group(1) if match else f"{args.save_path}.pth"

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

    # Send email notification if requested
    if args.email:
        subject = f"Training Complete: {args.model.upper()} model"
        body = f"""Training and testing completed successfully.

Model: {args.model.upper()}
Epochs: {args.epochs}
Learning Rate: {args.lr}
Batch Size: {args.batch_size}
Scheduler: {args.scheduler}
Weights: {weights_path}
"""
        if args.num_rotations > 0:
            body += f"Data Augmentation: {args.num_rotations} rotations\n"
        send_email(subject, body)


if __name__ == "__main__":
    main()
