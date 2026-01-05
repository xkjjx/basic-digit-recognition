import argparse
import torch
from common import create_cnn_model, create_mlp_model

MODEL_CONFIG = {
    "cnn": {
        "create": create_cnn_model,
        "input_shape": (1, 1, 28, 28),  # batch, channels, height, width
    },
    "mlp": {
        "create": create_mlp_model,
        "input_shape": (1, 28, 28),  # batch, height, width
    },
}


def convert_to_onnx(weights_path, output_path, model_type):
    """Convert PyTorch weights to ONNX format."""
    config = MODEL_CONFIG[model_type]
    model = config["create"]()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    
    dummy_input = torch.randn(*config["input_shape"])
    
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
    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["cnn", "mlp"],
        default="cnn",
        help="Model architecture: 'cnn' or 'mlp' (default: cnn)",
    )
    args = parser.parse_args()
    
    if args.format == "onnx":
        convert_to_onnx(args.weights_file, args.output, args.model)
    elif args.format == "json":
        convert_to_json(args.weights_file, args.output)


if __name__ == "__main__":
    main()
