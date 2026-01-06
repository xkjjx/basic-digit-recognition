## About

MNIST digit recognition with MLP and CNN models.

## Structure

```
train/          # Training scripts (train_mlp.py, train_cnn.py)
visualize/      # Model visualization generators
utils/          # Format conversion (pth → json/onnx)
data/           # MNIST dataset
weights/        # Saved model weights
visualizations/ # Generated analysis outputs
common.py       # Shared utilities
test.py         # Model evaluation
```

## Usage

**Train:** `python -m train.train_mlp` or `python -m train.train_cnn`  
**Test:** `python test.py`  
**Visualize:** `python -m visualize.visualize_cnn`  
**Convert:** `python -m utils.change_format`
