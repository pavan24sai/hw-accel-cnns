#include "MaxPoolingLayer.h"

// Constructor
MaxPoolingLayer::MaxPoolingLayer(const std::string& name, int poolSize, int stride) :
    Layer(name), poolSize(poolSize), stride(stride) {
}

// Forward pass
Tensor3D MaxPoolingLayer::forward(const Tensor3D& input) {
    int channels = input.getDepth();
    int inputHeight = input.getHeight();
    int inputWidth = input.getWidth();

    int outputHeight = (inputHeight - poolSize) / stride + 1;
    int outputWidth = (inputWidth - poolSize) / stride + 1;

    Tensor3D output(channels, outputHeight, outputWidth);

    for (int c = 0; c < channels; c++) {
        for (int row = 0; row < outputHeight; row++) {
            for (int col = 0; col < outputWidth; col++) {
                float maxVal = -std::numeric_limits<float>::max();
                for (int i = 0; i < poolSize; i++) {
                    for (int j = 0; j < poolSize; j++) {
                        int h = row * stride + i;
                        int w = col * stride + j;
                        maxVal = std::max(maxVal, input.at(c, h, w));
                    }
                }
                output.at(c, row, col) = maxVal;
            }
        }
    }

    return output;
}