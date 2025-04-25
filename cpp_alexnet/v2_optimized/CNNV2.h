#pragma once

#include "Layer.h"
#include "ConvolutionalLayerV2.h"
#include "MaxPoolingLayer.h"
#include "FullyConnectedLayer.h"
#include "Tensor3D.h"
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

/**
 * Updated CNN class that uses the optimized convolutional layer.
 */
class CNNV2 {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    CNNV2();

    // Load weights for all layers
    bool loadWeights(const std::string& basePath);

    // Forward pass through the entire network
    std::vector<float> forward(const Tensor3D& input);

    // Get top-k predictions
    std::vector<std::pair<int, float>> getTopKPredictions(const std::vector<float>& probabilities, int k);
};