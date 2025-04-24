#include "FullyConnectedLayer.h"

// Constructor
FullyConnectedLayer::FullyConnectedLayer(const std::string& name, int inputSize, int outputSize) :
    Layer(name), inputSize(inputSize), outputSize(outputSize),
    weights(outputSize, std::vector<float>(inputSize, 0.0f)),
    bias(outputSize, 0.0f) {
}

// Forward pass
Tensor3D FullyConnectedLayer::forward(const Tensor3D& input) {
    std::vector<float> flattenedInput;
    flattenedInput.reserve(input.getDepth() * input.getHeight() * input.getWidth());

    for (int d = 0; d < input.getDepth(); d++) {
        for (int h = 0; h < input.getHeight(); h++) {
            for (int w = 0; w < input.getWidth(); w++) {
                flattenedInput.push_back(input.at(d, h, w));
            }
        }
    }

    std::vector<float> outputVec(outputSize, 0.0f);
    for (int i = 0; i < outputSize; i++) {
        outputVec[i] = bias[i];
        for (int j = 0; j < inputSize; j++) {
            outputVec[i] += weights[i][j] * flattenedInput[j];
        }
        if (name != "fc8") {
            outputVec[i] = std::max(0.0f, outputVec[i]);
        }
    }

    Tensor3D output(1, 1, outputSize);
    for (int i = 0; i < outputSize; i++) {
        output.at(0, 0, i) = outputVec[i];
    }

    return output;
}

// Initialize weights
void FullyConnectedLayer::initializeWeights(float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            weights[i][j] = dist(gen);
        }
        bias[i] = dist(gen);
    }
}

// Load weights
bool FullyConnectedLayer::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    size_t weightsSize = outputSize * inputSize * sizeof(float);
    size_t biasSize = outputSize * sizeof(float);

    std::vector<float> weightsData(outputSize * inputSize);
    file.read(reinterpret_cast<char*>(weightsData.data()), weightsSize);

    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            weights[i][j] = weightsData[i * inputSize + j];
        }
    }

    file.read(reinterpret_cast<char*>(bias.data()), biasSize);
    return true;
}