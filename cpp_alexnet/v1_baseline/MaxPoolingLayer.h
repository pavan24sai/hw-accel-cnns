#pragma once

#include "Layer.h"
#include <limits>

/*
* Performs spatial downsampling by selecting the maximum value in each pooling window. This reduces the spatial dimensions 
* while preserving important features, making the network more computationally efficient and providing some translation invariance. 
* Unlike convolutional layers, pooling layers have no learnable parameters.
*/

class MaxPoolingLayer : public Layer {
private:
    int poolSize;
    int stride;

public:
    MaxPoolingLayer(const std::string& name, int poolSize, int stride);

    virtual Tensor3D forward(const Tensor3D& input) override;
};