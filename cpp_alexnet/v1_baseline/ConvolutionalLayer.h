#pragma once

#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include "Layer.h"
#include <vector>
#include <random>
#include <fstream>
#include <iostream>

/*
* Implements the core spatial filtering operation in CNNs. This layer applies learned filters (kernels) to detect features in the 
* input by performing sliding window multiplication and accumulation operations. It handles padding to maintain spatial dimensions 
* and applies ReLU activation to introduce non-linearity. The main computational complexity of the network resides here.
*/

class ConvolutionalLayer : public Layer {
private:
    int inputChannels;
    int outputChannels;
    int kernelSize;
    int stride;
    int padding;
    Tensor3D weights;
    std::vector<float> bias;

public:
    ConvolutionalLayer(const std::string& name, int inputChannels, int outputChannels, int kernelSize, int stride, int padding = 0);

    virtual Tensor3D forward(const Tensor3D& input) override;
    void initializeWeights(float stddev = 0.01f);
    virtual bool loadWeights(const std::string& filename) override;
};

#endif // CONVOLUTIONALLAYER_H