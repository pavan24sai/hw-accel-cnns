#pragma once

#ifndef CNN_H
#define CNN_H

#include "Layer.h"
#include "ConvolutionalLayer.h"
#include "MaxPoolingLayer.h"
#include "FullyConnectedLayer.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

/*
* Orchestrates the complete network by assembling all layers in sequence and managing the data flow between them. 
* This class handles model initialization, weight loading, and provides methods for inference. It implements the forward pass 
* through all layers and processes the network output (applying softmax and identifying top predictions).
*/

class CNN {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unordered_map<std::string, std::pair<int, int>> layerSizes;

public:
    CNN();
    bool loadWeights(const std::string& basePath);
    void loadLayerMetadata(const std::string& filename);
    std::vector<float> forward(const Tensor3D& input);
    std::vector<std::pair<int, float>> getTopKPredictions(const std::vector<float>& probabilities, int k);
};

#endif // CNN_H