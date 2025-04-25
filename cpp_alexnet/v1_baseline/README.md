# Version 1: Baseline Implementation

## 1. Architecture

The implementation follows the original AlexNet architecture:

- **5 convolutional layers** with various kernel sizes and strides.
- **3 max pooling layers** for spatial downsampling.
- **3 fully connected layers** for classification.
- **ReLU activations** throughout the network.
- **Final softmax layer** for class probabilities.

## 2. Class-Based Implementation
This is the first version of the AlexNet implementation. These are all the classes included as part of this:

### Tensor3D
- Core data structure for 3D feature maps and weights.
- Handles memory layout and access patterns.
- Supports efficient indexing via `at(d, h, w)` method.

### Layer (Base Class)
- Abstract interface for all network layers.
- Defines common methods: `forward()` and `loadWeights()`.
- Enables uniform layer processing in the network.

### ConvolutionalLayer
- Implements sliding window convolution operations.
- Handles padding, stride, and feature transformations.
- Applies ReLU activation after convolution.
- Contains learnable weights and biases.

### MaxPoolingLayer
- Performs spatial downsampling via max operation.
- Reduces dimensionality while preserving important features.
- No learnable parameters.

### FullyConnectedLayer
- Implements traditional neural network layers.
- Transforms spatial features into classification outputs.
- Contains majority of model parameters.

### CNN
- Main class that assembles the complete network.
- Manages layer initialization and weight loading.
- Orchestrates the forward pass for inference.
- Processes outputs with softmax for final predictions.