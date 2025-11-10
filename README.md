# CNN-Tensorflow

A deep learning project implementing Convolutional Neural Networks (CNNs) for Fashion-MNIST image classification using TensorFlow/Keras. This project contains multiple implementations ranging from simple CNNs to complex architectures with advanced techniques like batch normalization, dropout, and residual connections.

## Overview

This project implements various CNN architectures to classify Fashion-MNIST images into 10 categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Dataset

The project uses the **Fashion-MNIST** dataset, which consists of:
- **Training set**: 60,000 grayscale images (28×28 pixels)
- **Test set**: 10,000 grayscale images (28×28 pixels)
- **Classes**: 10 different clothing categories

The dataset is loaded from `.npy` files:
- `fashion_mnist_train_images.npy`
- `fashion_mnist_train_labels.npy`
- `fashion_mnist_test_images.npy`
- `fashion_mnist_test_labels.npy`

## Project Structure

```
CNN-Tensorflow/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── homework5.pdf             # Assignment instructions
├── ques_4_template.py        # Neural network from scratch (manual implementation)
├── ques_5_a.py              # Complex CNN with advanced techniques
└── ques5.py                 # Simple CNN implementation
```

## Files Description

### `ques5.py`
A simple CNN implementation using TensorFlow/Keras with:
- Single convolutional layer (64 filters, 3×3 kernel)
- Max pooling layer
- ReLU activation
- Two fully connected layers (1024 and 10 units)
- Model checkpointing for saving best weights

**Key Features:**
- Basic CNN architecture
- Model checkpointing
- Visualization of predictions on test images

### `ques_5_a.py`
A sophisticated CNN architecture with advanced deep learning techniques:

**Architecture:**
- Multiple convolutional layers with increasing filter sizes (32 → 64 → 128 → 256)
- Batch normalization after each convolutional layer
- Dropout layers (0.4) for regularization
- Residual connection (Add layer) combining features from different depths
- Dense layers with batch normalization and dropout
- Adam optimizer with learning rate 0.001

**Key Features:**
- Deep CNN with 11+ convolutional layers
- Batch normalization for stable training
- Dropout for preventing overfitting
- Residual connections
- Comprehensive visualization of predictions

### `ques_4_template.py`
A neural network implementation from scratch (without TensorFlow/Keras) featuring:

**Features:**
- Manual forward and backward propagation
- Custom max pooling implementation
- ReLU and softmax activations
- Stochastic Gradient Descent (SGD) optimizer
- L2 regularization
- Hyperparameter tuning functionality
- PCA-based visualization of SGD trajectory

**Key Components:**
- `forward_prop()`: Manual forward propagation
- `back_prop()`: Manual backpropagation
- `train()`: Training loop with SGD
- `findBestHyperparameters()`: Grid search for hyperparameter optimization
- `plotSGDPath()`: 3D visualization of optimization trajectory

## Requirements

### Dependencies
```
tensorflow>=2.0
numpy
matplotlib
scipy
scikit-learn
pickle (built-in)
```

### Installation

Install the required packages using pip:

```bash
pip install tensorflow numpy matplotlib scipy scikit-learn
```

## Usage

### Running the Simple CNN (`ques5.py`)

```bash
python ques5.py
```

This will:
1. Load the Fashion-MNIST dataset
2. Preprocess the data (normalization, reshaping, one-hot encoding)
3. Train a simple CNN model
4. Evaluate on the test set
5. Visualize predictions on random test images
6. Save the best model weights to `model.weights.best.hdf5`

### Running the Complex CNN (`ques_5_a.py`)

```bash
python ques_5_a.py
```

This will:
1. Load and preprocess the Fashion-MNIST dataset
2. Build a deep CNN with advanced techniques
3. Train for 10 epochs with validation
4. Evaluate on test set
5. Display predictions on 15 random test images

### Running the Manual Implementation (`ques_4_template.py`)

```bash
python ques_4_template.py
```

**Note:** This file contains code for manual neural network implementation and may require modifications to run depending on your specific use case. It includes:
- Custom weight initialization
- Manual gradient computation
- Hyperparameter search capabilities
- SGD trajectory visualization

## Model Architectures

### Simple CNN (ques5.py)
```
Input (28×28×1)
  ↓
Conv2D (64 filters, 3×3) → MaxPooling2D → ReLU
  ↓
Flatten
  ↓
Dense (1024) → ReLU
  ↓
Dense (10) → Softmax
```

### Complex CNN (ques_5_a.py)
```
Input (28×28×1)
  ↓
Conv Block 1: 32 filters (3×3, 3×3, 5×5 stride=2) + BatchNorm + Dropout(0.4)
  ↓
Conv Block 2: 64 filters (3×3, 3×3, 5×5 stride=2) + BatchNorm + Dropout(0.4)
Conv Block 3: 128→256 filters + BatchNorm + Dropout(0.4)
  ↓
Residual Connection (Add)
  ↓
Conv (256 filters, 7×7 stride=2) + BatchNorm
  ↓
Flatten
  ↓
Dense (512) + BatchNorm + Dropout(0.5)
  ↓
Dense (10) → Softmax
```

## Results

The models are trained to classify Fashion-MNIST images. Performance metrics (accuracy, loss) are printed during and after training. The complex CNN (`ques_5_a.py`) typically achieves better performance due to its deeper architecture and regularization techniques.

## Visualization

Both `ques5.py` and `ques_5_a.py` include visualization code that displays:
- Random test images with their predicted and true labels
- Color-coded predictions (green for correct, red for incorrect)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Rohin Siddhartha

## Notes

- Ensure the Fashion-MNIST `.npy` files are in the same directory as the Python scripts
- The validation set is created by splitting 5,000 samples from the training data
- Images are normalized to [0, 1] range by dividing by 255
- Labels are one-hot encoded for multi-class classification
