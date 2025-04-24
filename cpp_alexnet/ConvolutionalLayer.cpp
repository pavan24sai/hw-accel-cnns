#include "ConvolutionalLayer.h"

// Constructor
ConvolutionalLayer::ConvolutionalLayer(const std::string& name,
    int inputChannels,
    int outputChannels,
    int kernelSize,
    int stride,
    int padding) :
    Layer(name),
    inputChannels(inputChannels),
    outputChannels(outputChannels),
    kernelSize(kernelSize),
    stride(stride),
    padding(padding),
    weights(outputChannels, inputChannels, kernelSize* kernelSize),
    bias(outputChannels, 0.0f) {
}

// Forward pass
Tensor3D ConvolutionalLayer::forward(const Tensor3D& input) {
    int inputHeight = input.getHeight();
    int inputWidth = input.getWidth();

    // Calculate output dimensions
    int outputHeight = ((inputHeight + 2 * padding - kernelSize) / stride) + 1;
    int outputWidth = ((inputWidth + 2 * padding - kernelSize) / stride) + 1;

    // Create output tensor
    Tensor3D output(outputChannels, outputHeight, outputWidth, 0.0f);

    // Perform convolution
    for (int row = 0; row < outputHeight; row++) {
        for (int col = 0; col < outputWidth; col++) {
            for (int to = 0; to < outputChannels; to++) {
                output.at(to, row, col) = bias[to];
                for (int ti = 0; ti < inputChannels; ti++) {
                    for (int i = 0; i < kernelSize; i++) {
                        for (int j = 0; j < kernelSize; j++) {
                            int inputRow = stride * row + i - padding;
                            int inputCol = stride * col + j - padding;
                            if (inputRow >= 0 && inputRow < inputHeight &&
                                inputCol >= 0 && inputCol < inputWidth) {
                                int weightIdx = i * kernelSize + j;
                                output.at(to, row, col) +=
                                    weights.at(to, ti, weightIdx) *
                                    input.at(ti, inputRow, inputCol);
                            }
                        }
                    }
                }
                output.at(to, row, col) = std::max(0.0f, output.at(to, row, col));
            }
        }
    }

    return output;
}

// Initialize weights
void ConvolutionalLayer::initializeWeights(float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int to = 0; to < outputChannels; to++) {
        for (int ti = 0; ti < inputChannels; ti++) {
            for (int k = 0; k < kernelSize * kernelSize; k++) {
                weights.at(to, ti, k) = dist(gen);
            }
        }
        bias[to] = dist(gen);
    }
}

// Load weights
bool ConvolutionalLayer::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open weight file: " << filename << std::endl;
        return false;
    }

    // Use size_t for intermediate calculations to prevent overflow
    size_t weightsSize = static_cast<size_t>(outputChannels) *
        static_cast<size_t>(inputChannels) *
        static_cast<size_t>(kernelSize) *
        static_cast<size_t>(kernelSize) *
        sizeof(float);
    size_t biasSize = static_cast<size_t>(outputChannels) * sizeof(float);

    std::vector<float> weightsData(outputChannels * inputChannels * kernelSize * kernelSize);
    file.read(reinterpret_cast<char*>(weightsData.data()), weightsSize);

    for (int to = 0; to < outputChannels; to++) {
        for (int ti = 0; ti < inputChannels; ti++) {
            for (int k = 0; k < kernelSize * kernelSize; k++) {
                int idx = (to * inputChannels * kernelSize * kernelSize) +
                    (ti * kernelSize * kernelSize) + k;
                weights.at(to, ti, k) = weightsData[idx];
            }
        }
    }

    file.read(reinterpret_cast<char*>(bias.data()), biasSize);
    return true;
}