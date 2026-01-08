## About

MNIST digit recognition with MLP and CNN models.

## Structure

```
train/              # Training scripts (train_mlp.py, train_cnn.py)
visualize/          # Model visualization generators
utils/              # Format conversion (pth → json/onnx)
data/               # MNIST dataset
weights/            # Saved model weights
visualizations/     # Generated analysis outputs
common.py           # Shared utilities
test.py             # Model evaluation
train_and_test.py   # Unified train + test script
```

## Usage

**Train & Test:**
```bash
uv run train_and_test.py --model cnn
uv run train_and_test.py --model mlp --epochs 50 --lr 0.0005
```

**Train only:**
```bash
uv run -m train.train_cnn
uv run -m train.train_mlp --scheduler step --batch-size 128
```

**Test:** `uv run test.py weights/cnn_weights.pth --model-type cnn`
**Visualize:** `uv run -m visualize.visualize_cnn`
**Convert:** `uv run -m utils.change_format`

## Training Options

Both training scripts support:
- `--lr` - Learning rate (default: 0.001)
- `--epochs` - Training epochs (default: 100)
- `--batch-size` - Batch size (default: 64)
- `--scheduler` - LR scheduler: none, step, cosine, exponential, onecycle (default: cosine)
- `--format` - Output format: pth, onnx, json (default: pth)

GPU acceleration via CUDA or MPS (Apple Silicon) is automatic.
