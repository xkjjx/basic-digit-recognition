import argparse
import torch
from common import create_model


def convert_to_onnx(weights_path, output_path):
    """Convert PyTorch weights to ONNX format."""
    model = create_model()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    
    # Create dummy input matching expected shape (batch, height, width)
    dummy_input = torch.randn(1, 28, 28)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,  # Use legacy exporter (more stable with Python 3.14)
    )
    print(f"Converted to ONNX: {output_path}")


def convert_to_json(weights_path, output_path):
    """Convert PyTorch weights to JSON format (raw weights as nested lists)."""
    import json
    
    state_dict = torch.load(weights_path, weights_only=True)
    
    # Convert tensors to nested Python lists
    json_dict = {}
    for key, tensor in state_dict.items():
        json_dict[key] = tensor.tolist()
    
    with open(output_path, "w") as f:
        json.dump(json_dict, f)
    print(f"Converted to JSON: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model weights to other formats"
    )
    parser.add_argument(
        "weights_file",
        type=str,
        help="Path to the PyTorch weights file (.pth)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["onnx", "json"],
        required=True,
        help="Output format: 'onnx' (language-agnostic, recommended) or 'json' (raw weights)",
    )
    args = parser.parse_args()
    
    if args.format == "onnx":
        convert_to_onnx(args.weights_file, args.output)
    elif args.format == "json":
        convert_to_json(args.weights_file, args.output)


if __name__ == "__main__":
    main()

