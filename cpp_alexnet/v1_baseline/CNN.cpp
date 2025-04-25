#include "CNN.h"

// Constructor
CNN::CNN() {
    // Create layers based on the actual AlexNet architecture from PyTorch

    // Convolutional Layers (with padding)
    layers.push_back(std::make_unique<ConvolutionalLayer>("conv1", 3, 64, 11, 4, 2));     // Output: 55x55
    layers.push_back(std::make_unique<MaxPoolingLayer>("pool1", 3, 2));                   // Output: 27x27

    layers.push_back(std::make_unique<ConvolutionalLayer>("conv2", 64, 192, 5, 1, 2));    // Output: 27x27
    layers.push_back(std::make_unique<MaxPoolingLayer>("pool2", 3, 2));                   // Output: 13x13

    layers.push_back(std::make_unique<ConvolutionalLayer>("conv3", 192, 384, 3, 1, 1));   // Output: 13x13

    layers.push_back(std::make_unique<ConvolutionalLayer>("conv4", 384, 256, 3, 1, 1));   // Output: 13x13

    layers.push_back(std::make_unique<ConvolutionalLayer>("conv5", 256, 256, 3, 1, 1));   // Output: 13x13
    layers.push_back(std::make_unique<MaxPoolingLayer>("pool5", 3, 2));                   // Output: 6x6

    // Fully Connected Layers - using the exact dimensions from PyTorch model
    int featMapSize = 6 * 6 * 256;  // 9216
    layers.push_back(std::make_unique<FullyConnectedLayer>("fc6", featMapSize, 4096));
    layers.push_back(std::make_unique<FullyConnectedLayer>("fc7", 4096, 4096));
    layers.push_back(std::make_unique<FullyConnectedLayer>("fc8", 4096, 1000));  // 1000 classes for ImageNet
}

// Load weights from binary files
bool CNN::loadWeights(const std::string& basePath) {
    bool success = true;

    // Load metadata if available
    loadLayerMetadata(basePath + "/network_metadata.txt");

    for (auto& layer : layers) {
        // Skip pooling layers which don't have weights
        if (dynamic_cast<MaxPoolingLayer*>(layer.get())) {
            continue;
        }

        std::string layerName = layer->getName();
        std::string filename = basePath + "/" + layerName + "_combined.bin";

        std::cout << "Loading weights for layer: " << layerName << " from " << filename << std::endl;

        // Load weights
        if (!layer->loadWeights(filename)) {
            std::cerr << "Failed to load weights for layer: " << layerName << std::endl;
            success = false;
        }
    }

    return success;
}

// Load network metadata (optional, helps with weight loading)
void CNN::loadLayerMetadata(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Metadata file not found: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string layerName;
        int outSize, inSize;

        if (iss >> layerName >> outSize >> inSize) {
            layerSizes[layerName] = { inSize, outSize };
            std::cout << "Loaded metadata for " << layerName
                << ": in=" << inSize << ", out=" << outSize << std::endl;
        }
    }
}

// Forward pass through the entire network
std::vector<float> CNN::forward(const Tensor3D& input) {
    Tensor3D current = input;

    // Pass through each layer
    for (const auto& layer : layers) {
        std::cout << "Processing layer: " << layer->getName() << std::endl;
        current = layer->forward(current);

        // Print output dimensions for debugging
        std::cout << "  Output shape: [" << current.getDepth() << ", "
            << current.getHeight() << ", " << current.getWidth() << "]" << std::endl;
    }

    // Extract output probabilities from final layer
    const auto& outputData = current.getData();

    // Apply softmax to get class probabilities
    std::vector<float> probabilities(outputData.size());
    float maxVal = *std::max_element(outputData.begin(), outputData.end());
    float sumExp = 0.0f;

    for (size_t i = 0; i < outputData.size(); i++) {
        probabilities[i] = std::exp(outputData[i] - maxVal);  // Subtract max for numerical stability
        sumExp += probabilities[i];
    }

    for (size_t i = 0; i < probabilities.size(); i++) {
        probabilities[i] /= sumExp;
    }

    return probabilities;
}

// Get top-k predictions and map to class labels
std::vector<std::pair<int, float>> CNN::getTopKPredictions(const std::vector<float>& probabilities, int k) {
    std::vector<std::pair<int, float>> idxProb;
    for (size_t i = 0; i < probabilities.size(); i++) {
        idxProb.push_back({ i, probabilities[i] });
    }

    // Sort by probability (descending)
    std::partial_sort(idxProb.begin(), idxProb.begin() + k, idxProb.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // Return top-k
    return std::vector<std::pair<int, float>>(idxProb.begin(), idxProb.begin() + k);
}